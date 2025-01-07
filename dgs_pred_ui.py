"""
A Web Ui for the DGS Price Prediction Model.
- Given a list of stock symbols
- Check of the existence of existing models for the symbols
- If the Symbols does not have models, train a model for the symbol
  - Generate a predictions for the symbol
- If the symbol has a model, load the model and generate predictions

Each element (buttons, input_text, etc) are held in the session_state object.
When creating a key for the element, prefix the key with b_ for buttons, t_ for text_input, etc.,
to make it easier to identify the type of element.
When creating one's own session state objects, prefix flags with ss_f, and dataframes with ss_df,
for additional clarity.
Additionally, it is good practice to place session state names into variables to avoid typos,
and quoted strings all over the code.

Note: All local variable are initialized in on .reset('app') method.
      Keep all required variables in the session_state object.
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import logging
import dill
import shutil
import yfinance as yf
import futureproof as fp
import hashlib
import re
import io
import concurrent.futures as cf
import objgraph

import line_profiler
from line_profiler import profile, LineProfiler

from datetime import datetime, timedelta
from pricepredict import PricePredict
from io import StringIO
from streamlit_js_eval import streamlit_js_eval

# Change working directory to the ~/workspace/pricepredict directory
os.chdir('/home/dsidlo/workspace/pricepredict')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Log to a file...
logging.basicConfig(filename='dgs_pred_ui.log', level=logging.DEBUG)


# Session State Variables
# DataFrames...
ss_DfSym = 'df_symbols'             # The main display DataFrame
ss_AllDfSym = 'all_df_symbols'      # Cache of all symbols
ss_SymDpps_d = 'sym_dpps_d'         # Daily PricePredict objects list
ss_SymDpps_w = 'sym_dpps_w'         # Weekly PricePredict objects list
# Buttons...
ss_bAddNewSyms = 'bAddNewSyms'
ss_bCancelTglFaves = 'ss_bCancelTglFaves'
ss_bToggleGroups = 'ss_bToggleGroups'
ss_bCancelRmSyms = 'cancel_rm_syms'
ss_bRemoveChosenSyms = 'remove_chosen_syms'
ss_forceOptimization = 'force_optimization'
# Flags...
ss_fDf_symbols_updated = 'st_df_symbols_updated'
ss_fRmChosenSyms = 'Remove Chosen Symbols Flag'
ss_forceTraining = "cbForceTraining"
ss_forceAnalysis = "cbForceAnalysis"
ss_fRemoveNewSyms = 'remove_new_sym'
ss_fYesRemoveNewSyms = 'yes_remove_new_sym'
# Variables for Grouping Operations...
ss_GroupsList = 'GroupsList'  # Current Groups List
tiNewGrp = 'tiNewGrp'  # Text input for adding a new group
bRremoveGroup = 'bRremoveGroup'  # Button for removing a group
sbRremoveGroup = 'sbRremoveGroup'  # Selectbox for removing a group
sbToggleGrp = 'sbToggleGrp'  # Selectbox for toggling a group
ss_val_addNewSyms = 'val_addNewSyms'

# Symbol under review
img_sym = None
# PP objects Daily
sym_dpps = None
sym_dpps_w = None

# Directory for gui files
gui_data = './gui_data/'
# Save/Restore file for the all_df_symbols DataFrame
guiAllSymbolsCsv = f'{gui_data}gui_all_symbols.csv'
# Pickle files for the PP objects
dill_sym_dpps_d = f'{gui_data}sym_dpps.dil'
dill_sym_dpps_w = f'{gui_data}sym_dpps_w.dil'
dillbk_sym_dpps_d = f'{gui_data}sym_dpps.dil.bk'
dillbk_sym_dpps_w = f'{gui_data}sym_dpps_w.dil.bk'
# JSON file for the optimized hyperparameters
opt_hyperparams = f'{gui_data}ticker_bopts.json'

# Direcoty paths...
model_dir = './models/'
chart_dir = './charts/'
preds_dir = './predictions/'

import_cols_simple = 2
import_cols_full = 12

# ======================================================================
# Main Window ==========================================================
# The main window displays the charts and data for the selected symbol.
# ======================================================================
@profile
def  main(message):

    logger.info(f"*** {message} ***")

    # Clear the session state
    with st.empty():
        pass

    if ss_AllDfSym not in st.session_state.keys():
        # This is only needed for bootstrapping the development of this app.
        # When the app is fully developed, we load the symbols from a file or database.
        logger.info("Initializing df_symbols")
        if os.path.isfile(guiAllSymbolsCsv):
            # --------------------------------------------------------
            # Startup Initialization: Load the symbols from the file
            # --------------------------------------------------------
            with st.spinner("# Loading Symbols and PricePredict Objects..."):
                # Load the symbols from the file
                logger.debug(f"Loading all_df_symbols from {guiAllSymbolsCsv} from file {guiAllSymbolsCsv}")
                st.session_state[ss_AllDfSym] = pd.read_csv(guiAllSymbolsCsv, header='infer')
                # Clear the Group field if it contains 'Imported' or 'Added'
                st.session_state[ss_AllDfSym]['Groups'] = st.session_state[ss_AllDfSym]['Groups'].replace(['Imported', 'Added'], '')
                # Load pp daily objects
                logger.debug(f"Loading sym_dpps/w from {dill_sym_dpps_d} and {dillbk_sym_dpps_w}")
                st.session_state[ss_SymDpps_d], st.session_state[ss_SymDpps_w] = load_pp_objects(st)

        if ss_DfSym not in st.session_state:
            # Save the DataFrame to the session state
            all_df_symbols = st.session_state[ss_AllDfSym]
            df_symbols = all_df_symbols.copy()
            st.session_state[ss_DfSym] = all_df_symbols

        # Update the DataFrame with the latest data
        all_df_symbols = update_viz_data(st, st.session_state[ss_AllDfSym])

        # Get the list of groups from the DataFrame
        st.session_state[ss_GroupsList] = all_df_symbols['Groups'].unique().tolist()
        # Replace nan values in st.session_state[ss_GroupsList] with ''
        st.session_state[ss_GroupsList] = [x if x == x else '' for x in st.session_state[ss_GroupsList]]
        # Sort the list of groups
        st.session_state[ss_GroupsList].sort()

        st.session_state[ss_AllDfSym] = all_df_symbols.copy()
        st.session_state[ss_DfSym] = all_df_symbols.copy()
        df_symbols = st.session_state[ss_DfSym]
        pass

    # ==============================================================
    # Side Bar =====================================================
    # The side bar controls most of what happens in the app.
    # ==============================================================
    with (st.sidebar):
        st.sidebar.title("*Price Prediction for Swing Trading*")
        st.checkbox("Interactive Charts", key='cbInteractiveCharts')
        # Add Groups ==========================================
        if ss_GroupsList not in st.session_state:
            st.session_state[ss_GroupsList] = []
        # Add Expander for adding and removing groupings
        # st.button("End Program", key='bEndProgram')
        # if st.session_state.bEndProgram:
        #     create_charts_dict(st)
        #     charts_cleanup(st, PricePredict.PeriodDaily)
        #     charts_cleanup(st, PricePredict.PeriodWeekly)
        #     st.success("Program Ended")
        #     st.stop()

        # -- Add Expander for adding a new symbol
        exp_sym = st.expander("Add/Remove Symbols", expanded=False)
        exp_sym.text('------------------------------------------------------------')
        st.session_state['exp_sym'] = exp_sym
        col1, col2 = exp_sym.columns(2)
        # -- Add a text input for adding a new symbol
        if 'new_sym_value' not in locals():
            new_sym_value = ''
        bAddNewSyms = col1.button("Add New Symbols", key="ss_bAddNewSyms")
        ti_addNewSyms = col2.text_input("Add New Symbols",
                                           key="new_sym", label_visibility='collapsed',
                                           value='', placeholder="Enter a new symbol")
        # -- Add a file uploader button for importing symbols
        bt_import_syms = "import_syms"
        imported_data = b''
        exp_sym.file_uploader("Import Symbols", key="import_syms",
                                type='csv')
        # -- Add a button for Optimize Models
        opt_cols = exp_sym.columns(2)
        opt_cols[0].button("Optimize Models", key="b_optmodels")
        opt_cols[1].checkbox("Force optimization", key=ss_forceOptimization)
        exp_sym.text('------------------------------------------------------------')
        # -- Add a button for initiating removing the imported symbols
        col_rm_syms1, col_rm_syms2 = exp_sym.columns(2)
        col_rm_syms1.button("Remove Added Symbols", key="remove_imp_sym")
        if (hasattr(st.session_state, ss_fDf_symbols_updated) and st.session_state[ss_fDf_symbols_updated]
            # Indicate successful removal of New symbols
            and st.session_state['imp_sums_removed'] > 0):
            exp_sym.success(f'**[{st.session_state["imp_sums_removed"]}] New Symbols Removed.**', icon='👍')
            st.session_state[ss_fDf_symbols_updated] = False
        # -- Add a button for adding the new symbol
        col_rm_syms2.button("Choose Symbols to Remove", key="bRemove_sym")
        # -- Add a button for process the current symbols
        col1, col2 = exp_sym.columns(2)
        col1.button("Analyze Current Symbols", key="process_syms")
        col2.checkbox("Force Training", key=ss_forceTraining)
        col2.checkbox("Force Analysis", key=ss_forceAnalysis)
        if st.session_state.process_syms:
            with exp_sym, st.spinner("Analyzing Symbols (Please be patient)..."):
                prog_bar = exp_sym.progress(0, "Analyzing Symbols")
                analyze_symbols(st, prog_bar, st.session_state[ss_DfSym],
                                force_training=st.session_state[ss_forceTraining],
                                force_analysis=st.session_state[ss_forceAnalysis])

        elif 'prog_bar' in locals():
            prog_bar.empty()

        # -- Handle the import of symbols
        if st.session_state.import_syms is not None and not st.session_state.bRemove_sym:
            import_symbols(st, exp_sym)

        st.sidebar.add_rows = exp_sym

        # -- Handle add new symbols text input
        if bAddNewSyms and ti_addNewSyms != '':
            # Split on new_sym on spaces, colon, semicolon and commas
            new_syms = ti_addNewSyms
            new_syms = re.split(r'[\s,;]+', new_syms)
            if new_syms[0] != '':
                add_new_symbols(st, exp_sym, new_syms)
                st.session_state[ss_DfSym] = update_viz_data(st, st.session_state[ss_AllDfSym])

            # Save all_df_symbols
            all_df_symbols = st.session_state[ss_AllDfSym]
            df_symbols = st.session_state[ss_DfSym]
            merge_and_save(all_df_symbols, df_symbols)
            ti_addNewSyms.replace(ti_addNewSyms, '')

        # -- Handle the Optimize Models button
        if st.session_state.b_optmodels:
            with exp_sym, st.spinner("Optimizing Models (Please be patient)..."):
                pb_opt_hparams = exp_sym.progress(0, "Optimize Symbols Hyperparameters")
                optimize_hparams(st, pb_opt_hparams)
            pb_store_objs = exp_sym.progress(0, "Storing PricePredict Objects")
            store_pp_objects(st, pb_store_objs)

        # ********* Add/Remove Groups *********
        exp_grps = st.expander("Add/Remove Groups", expanded=False)
        st.session_state['exp_grps'] = exp_grps
        col_add_grp1, col_add_grp2 = exp_grps.columns(2)
        col_add_grp1.button("Add New Groups", key='bAddGroups')
        ti_addNewGrp = col_add_grp2.text_input("Add New Group", label_visibility='collapsed',
                                               key=tiNewGrp, value='', placeholder="Enter a new Group")
        if hasattr(st.session_state, 'bAddGroups'):
            if st.session_state.bAddGroups and st.session_state.tiNewGrp != '':
                new_grps = ti_addNewGrp
                new_grps = re.split(r'[\s,;]+', new_grps)
                added_grps = []
                if hasattr(st.session_state, 'ss_GroupsList'):
                    curr_grps = st.session_state[ss_GroupsList]
                else:
                    curr_grps = []
                if len(new_grps) > 0:
                    for grp in new_grps:
                        if grp != '' and grp not in curr_grps:
                            # Remove the Group from the Groups List
                            st.session_state[ss_GroupsList].append(grp)
                            added_grps.append(grp)
                    st.session_state[ss_GroupsList] = list(set(st.session_state[ss_GroupsList]))
                    st.session_state[ss_GroupsList].sort()
                else:
                    info_txt = "No new groups added."
                if len(added_grps) > 0:
                    info_txt = f"Added Groups: {added_grps}"
                exp_grps.info(info_txt)

        col_rm_grp1, col_rm_grp2 = exp_grps.columns(2)
        col_rm_grp1.button("Remove Group", key=bRremoveGroup)
        col_rm_grp2.selectbox("Remove Group", label_visibility='collapsed',
                               options=st.session_state[ss_GroupsList], key=sbRremoveGroup)
        # Remove Groups ==========================================
        if hasattr(st.session_state, bRremoveGroup) and st.session_state.bRremoveGroup:
            logger.debug(f"Remove Group: {st.session_state.sbRremoveGroup}")
            if st.session_state.sbRremoveGroup in st.session_state[ss_GroupsList]:
                st.session_state[ss_GroupsList].remove(st.session_state.sbRremoveGroup)
                st.session_state[ss_GroupsList] = list(set(st.session_state[ss_GroupsList]))
                st.session_state[ss_GroupsList].sort()
                exp_grps.info(f"Removed Group: {st.session_state.sbRremoveGroup}")
            else:
                exp_grps.info(f"Group [{st.session_state.sbRremoveGroup}] not found.")

            # Warn: Symbols with this group will have the group cleared.
            # - Use dialog box to confirm the removal of the group, or choose another group to replace it.

        # -- Create a Column Container for symbol operations and the symbol list
        col_tgl_grp1, col_tgl_grp2 = exp_grps.columns(2)
        # -- Add a button for initiating toggle favorites
        # Init the "Toggle Groups" state flag.
        col_tgl_grp1.button("Toggle Groups", key=ss_bToggleGroups)
        col_tgl_grp2.selectbox("Set Group", label_visibility='collapsed',
                               options=st.session_state[ss_GroupsList],
                               key=sbToggleGrp)
        # Init the "Remove Chosen Symbols" state flag.
        if ss_fRmChosenSyms not in st.session_state:
            st.session_state[ss_fRmChosenSyms] = False
        rm_chosen_syms = st.session_state[ss_fRmChosenSyms]

        # -- Add Expander for filtering the symbols
        exp_sort_filter = st.expander("Filter", expanded=False)
        # -- Add a dropdown for filtering the symbols
        exp_sort_filter.selectbox(
            # Add a dropdown for filtering the symbols
            "***- filter -***",
            ("All", "Up Trending", "Down Trending", "Flat", "New"),
            on_change=filter_symbols,
            # Make this elements value available via the st.session_state object.
            key="filter_sym"
        )

        # -- Create a Column Container for symbol operations and the symbol list
        sym_col = st.columns(1)

        # Single-Row / Mylti-Row Toggle :  =================
        if ((st.session_state.bRemove_sym or rm_chosen_syms)               # Remove Chosen Symbols
            or (st.session_state.sbToggleGrp is not None and st.session_state.sbToggleGrp.strip() != '')):   # Mark/Clear Favorites
            logger.info("2>>> Multi-Row on")
            # if st.session_state.bRemove_sym or rm_chosen_syms:   # Remove Chosen Symbols
            #     st.session_state.sbToggleGrp = True
            df_sel_mode = ["multi-row"]
        else:
            logger.info("2>>> Multi-Row off")
            df_sel_mode = ["single-row"]
        # =============================================================

        # Display the DataFrame of Symbols ============================
        df_sym = st_dataframe_widget(exp_sym, ss_DfSym, df_sel_mode, sym_col)
        # =============================================================

        # Action Buttons : Remove Chosen Symbols ======================
        img_sym = None
        if st.session_state.bRemove_sym or rm_chosen_syms:
            # Remove the selected symbol from the DataFrame
            logger.info("*** Remove Selected Symbol ***")
            st.session_state[ss_fRmChosenSyms] = True
            sym_col.append(st.button("Remove Chosen Symbols", key="remove_chosen_syms"))
            sym_col.append(st.button("Cancel Remove Symbols Operation", key="cancel_rm_syms"))
        # =============================================================

        # Action Buttons : Mark/Clear Favorites =======================
        elif st.session_state.sbToggleGrp is not None and st.session_state.sbToggleGrp != '':
            logger.info("*** Toggle Groups ***")
            sym_col.append(st.button("Cancel Toggle Group Operation", key=ss_bCancelTglFaves))
        # =============================================================

        # Display Chosen Symbol's Chart & Data ========================
        elif hasattr(df_sym.selection, 'rows') and len(df_sym.selection.rows) > 0:
            # We have a selected symbol so display the symbol's chart
            df_symbols = st.session_state[ss_DfSym]
            logger.info(df_sym.selection.rows[0])
            logger.info(df_symbols.Symbol[df_sym.selection.rows[0]])
            img_sym = df_symbols.Symbol[df_sym.selection.rows[0]]
        # =============================================================

        # Remove New Symbols =====================================
        if hasattr(st.session_state, ss_fRemoveNewSyms) and st.session_state.remove_imp_sym:
            dlg_rm_imp_syms()

        if hasattr(st.session_state, ss_fYesRemoveNewSyms) and st.session_state.yes_remove_imp_sym:
            if hasattr(st.session_state, ss_fDf_symbols_updated) and st.session_state[ss_fDf_symbols_updated]:
                st.session_state[ss_fDf_symbols_updated] = False
                # st.rerun()
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
        # =============================================================

        # Remove Chosen Symbols =======================================
        if hasattr(st.session_state, ss_bRemoveChosenSyms) and st.session_state.remove_chosen_syms:
            st.session_state[ss_fRmChosenSyms] = False
            logger.info('*** Removing Chosen Symbols ***')
            # This operation removes the selected symbols from the visual DataFrame
            dfSave_syms = False
            for i in df_sym.selection.rows:
                dfSave_syms = True
                # Turns off the Action Buttons
                # Get the selected symbol
                sym = st.session_state[ss_DfSym].iloc[i].Symbol
                # Remove the selected symbol from the all_df_symbols DataFrame
                all_df_symbols = st.session_state[ss_AllDfSym]
                all_df_symbols = all_df_symbols[all_df_symbols['Symbol'] != sym]
                # Remove the selected symbol from the sym_dpps_w and sym_dpps dictionaries
                if sym in st.session_state[ss_SymDpps_w]:
                    del st.session_state[ss_SymDpps_w][sym]
                if sym in st.session_state[ss_SymDpps_d]:
                    del st.session_state[ss_SymDpps_d][sym]
            st.session_state[ss_DfSym] = st.session_state[ss_DfSym].drop(df_sym.selection.rows)
            st.session_state[ss_AllDfSym] = st.session_state[ss_AllDfSym].drop(df_sym.selection.rows)
            # Save out the updated DataFrames and PricePredict objects
            all_df_symbols = st.session_state[ss_AllDfSym]
            df_symbols = st.session_state[ss_DfSym]
            if dfSave_syms:
                logger.debug(f"1> Saving all_df_symbols to {guiAllSymbolsCsv}")
                merge_and_save(all_df_symbols, df_symbols)
            # st.rerun()
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        if hasattr(st.session_state, ss_bCancelRmSyms) and st.session_state.cancel_rm_syms:
            st.session_state[ss_fRmChosenSyms] = False  # Turns off the Action Buttons
            logger.info('*** Canceled: Remove Chosen Symbols ***')
            st.rerun()
        # =====================================================

        # Toggle Favorites ====================================
        if hasattr(st.session_state, ss_bToggleGroups) and st.session_state.ss_bToggleGroups:
            group_fld = 'Groups'
            sym_fild = 'Symbol'
            fSaveDfSyms = False
            # This operation toggles the "f" flag in the Type field of the selected symbols
            for row in df_sym.selection.rows:
                logger.info(f"Before Toggle: Sym:[{st.session_state[ss_DfSym].Symbol[row]}]"
                            + " to [{st.session_state[ss_DfSym].loc[row, group_fld]}]")
                if st.session_state[ss_DfSym].loc[row, group_fld] == st.session_state.sbToggleGrp:
                    st.session_state[ss_DfSym].loc[row, group_fld] = ''
                    logger.debug(f"Toggled: Sym:[{st.session_state[ss_DfSym].loc[row, sym_fild]}]"
                                + " to [{st.session_state[ss_DfSym].loc[row, group_fld]}]")
                    fSaveDfSyms = True
                else:
                    st.session_state[ss_DfSym].loc[row, group_fld] = st.session_state.sbToggleGrp
                    logger.debug (f"Toggled: Sym:[{st.session_state[ss_DfSym].loc[row, sym_fild]}]"
                                + " to [{st.session_state[ss_DfSym].loc[row, group_fld]}]")
                    fSaveDfSyms = True

            # Save out the updated DataFrames and PricePredict objects
            if fSaveDfSyms:
                logger.debug(f"2> Saving all_df_symbols to {guiAllSymbolsCsv}")
                all_df_symbols = st.session_state[ss_AllDfSym]
                df_symbols = st.session_state[ss_DfSym]
                merge_and_save(all_df_symbols, df_symbols)

            logger.info('*** Toggle Favorites ***')
            st.rerun()  # Refresh the page

        if hasattr(st.session_state, ss_bCancelTglFaves) and st.session_state.ss_bCancelTglFaves:
                logger.info('*** Canceled: Remove Chosen Symbols ***')
                st.rerun()  # Refresh the page
        # =====================================================

    display_symbol_charts(interactive_charts=st.session_state.cbInteractiveCharts)

def merge_and_save(all_df_symbols, df_symbols):
    # Merge df_symbols with all_df_symbols
    for row in df_symbols.itertuples():
        # Get the index to the existing symbol
        idx = all_df_symbols.index[all_df_symbols.Symbol == row.Symbol]
        if idx is not None:
            all_df_symbols.loc[idx] = [row.Symbol, row.LongName,
                                       row.Groups, row.Trend,
                                       row.WklyPrdStg, row.DlyPrdStg,
                                       row.wTop10Coint, row.dTop10Coint,
                                       row.wTop10Corr, row.wTop10xCorr,
                                       row.dTop10Corr, row.dTop10xCorr]
        elif len(idx) == 0:
            # Add the symbol to the DataFrame
            all_df_symbols.loc[len(all_df_symbols)] = [row.Symbol, row.LongName,
                                                       row.Groups, row.Trend,
                                                       row.WklyPrdStg, row.DlyPrdStg,
                                                       row.wTop10Coint, row.dTop10Coint,
                                                       row.wTop10Corr, row.wTop10xCorr,
                                                       row.dTop10Corr, row.dTop10xCorr]
    all_df_symbols.sort_values("Symbol", inplace=True)
    all_df_symbols.reindex()
    st.session_state[ss_AllDfSym] = all_df_symbols
    # Save out the updated DataFrames and PricePredict objects
    if len(all_df_symbols.columns) == import_cols_full:
        # Sort all_df_symbols by Symbol
        logger.debug(f"3> Saving all_df_symbols to {guiAllSymbolsCsv}")
        all_df_symbols.to_csv(guiAllSymbolsCsv, index=False)
    else:
        logger.error(
            f"Error all_df_symbols has too many columns [{len(all_df_symbols)}]: {all_df_symbols.columns}")

    st.session_state[ss_AllDfSym] = all_df_symbols
    st.session_state[ss_DfSym] = df_symbols

def getTickerLongName(chk_ticker):
    ticker = None
    long_name = ''
    try:
        ticker_ = yf.Ticker(chk_ticker)
        if ticker_ is not None:
            ticker_data = yf.Ticker(chk_ticker).info
            ticker = chk_ticker
            if 'longName' in ticker_data:
                long_name = ticker_data['longName']
            elif 'info' in ticker_data:
                if 'shortName' in ticker_data['info']:
                    long_name = ticker_data['info']['shortName']
                else:
                    long_name = ''
    except Exception as e:
        logger.error(f"Error in getTickerLongNamer: {e}")
        ticker_ = []
    if ticker_ is None:
        ticker = None
        long_name = None

    return ticker, long_name


def st_dataframe_widget(exp_sym, ss_DfSym, df_sel_mode, sym_col):
    if (st.session_state.bRemove_sym
            or (hasattr(st.session_state, ss_fRmChosenSyms)
                and st.session_state[ss_fRmChosenSyms])
            or (st.session_state.sbToggleGrp is not None and st.session_state.sbToggleGrp.strip() != '')
            or df_sel_mode == "multi-row"):
        # This is used when removing symbols and marking/clearing favorites
        on_select = 'rerun'
        df_sel_mode = 'multi-row'
        logger.info("1>>> Multi-Row on")
    else:
        # on_select = display_symbol_charts
        on_select = 'rerun'
        df_sel_mode = 'single-row'
        logger.info("1>>> Multi-Row off")

    print(f' *** df_sel_mode: {df_sel_mode}, on_select: {on_select}')
    print(f' *** ss_fRmChosenSyms [{hasattr(st.session_state, ss_fRmChosenSyms)}]')
    # Fill null values in the Groups column with an empty string
    st.session_state[ss_DfSym]['Groups'] = st.session_state[ss_DfSym]['Groups'].fillna('')
    st.session_state[ss_DfSym] = st.session_state[ss_DfSym].ffill()
    st.session_state[ss_DfSym] = st.session_state[ss_DfSym].bfill()
    if 'st_dataframe.df_sym' in st.session_state:
        st.session_state['st_dataframe.df_sym'] = None
    # df_symbols = st.session_state[ss_DfSym].copy()
    # df_symbols = st.session_state[ss_DfSym]
    # df_symbols.reindex()
    # df_styled = df_symbols.style.format({'WklyPrdStg': '{:,.4f}',
    #                                      'DlyPrdStg': '{:,.4f}'})
    st.session_state[ss_DfSym].reindex()
    df_symbols = st.session_state[ss_DfSym].copy(deep=True)
    df_sym = st.dataframe(df_symbols, key="dfSymbols", height=600,
                          column_order=["Symbol", "Groups", "Trend",
                                        "WklyPrdStg", "DlyPrdStg"],
                          column_config={"Symbol": {"max_width": 5},
                                         "Groups": {"max_width": 4},
                                         "Trend": {"max_width": 5},
                                         "WklyPrdStg": {"max_width": 5},
                                         "DlyPrdStg": {"max_width": 5}},
                          selection_mode=df_sel_mode, on_select=on_select, hide_index=True)
    sym_col.append(df_sym.selection)
    st.session_state['st_dataframe.df_sym'] = df_sym
    # Display Chosen Symbol's Chart & Data ========================
    if hasattr(df_sym.selection, 'rows') and len(df_sym.selection.rows) > 0:
        # We have a selected symbol so display the symbol's chart
        st.session_state[ss_DfSym] = df_symbols.copy(deep=True)
        logger.info(df_sym.selection.rows[0])
        logger.info(df_symbols.Symbol[df_sym.selection.rows[0]])
        img_sym = df_symbols.Symbol[df_sym.selection.rows[0]]
    # =============================================================

    return df_sym


def filter_symbols():
    filter = st.session_state.filter_sym
    all_df_symbols = st.session_state[ss_AllDfSym]
    df_symbols = st.session_state[ss_DfSym]
    # Merge current symbols with the imported symbols, incase any have benn added
    for row in df_symbols.itertuples():
        # Get the index to the existing symbol
        idx = all_df_symbols.index[all_df_symbols.Symbol == row.Symbol].tolist()
        if len(idx) > 1:
            logger.error(f"Symbol {row.Symbol} is duplicated in the DataFrame")
            raise ValueError(f"Symbol {row.Symbol} is duplicated in the DataFrame")
        elif len(idx) == 0:
            # Add the symbol to the DataFrame
            all_df_symbols.loc[len(all_df_symbols)] = [row.Symbol, row.LongName,
                                                       row.Groups, row.Trend,
                                                       row.WklyPrdStg, row.DlyPrdStg,
                                                       row.wTop10Coint, row.wTop10xCoint,
                                                       row.wTop10Corr, row.wTop10xCorr,
                                                       row.dTop10Corr, row.dTop10xCorr]
        elif len(idx) == 1:
            # Update the Type field with the imported value
            all_df_symbols.loc[idx[0], "Groups"] = row.Groups
    # Update the session_state DataFrame with only the filtered symbols
    if filter == "All":
        df_symbols = all_df_symbols
    elif filter == "Favorites":
        # Make df_symbols the "Favorites" subset of all_df_symbols
        df_symbols = all_df_symbols[all_df_symbols.Groups == "f"]
    elif filter == "Up Trending":
        # Make df_symbols the "Up Trending" subset of all_df_symbols
        df_symbols = all_df_symbols[all_df_symbols.Trend == "u"]
    elif filter == "Down Trending":
        df_symbols = all_df_symbols[all_df_symbols.Trend == "d"]
    elif filter == "Flat":
        df_symbols = all_df_symbols[all_df_symbols.Trend == "f"]
    elif filter == "Imported":
        df_symbols = all_df_symbols[all_df_symbols.Groups == "i"]

    # Set the session_state DataFrame to the filtered DataFrame
    st.session_state[ss_DfSym] = df_symbols
    # Replace None in the Type field with an empty string
    all_df_symbols['Groups'] = all_df_symbols['Groups'].fillna('')
    # Save all_df_symbols to a file
    if len(all_df_symbols.columns) == import_cols_full:
        logger.debug(f"4> Saving all_df_symbols to {guiAllSymbolsCsv}")
        all_df_symbols.to_csv(guiAllSymbolsCsv, index=False)
    else:
        logger.error(f"Error all_df_symbols has too many columns [{len(all_df_symbols)}]: {all_df_symbols.columns}")


@st.dialog("Remove Chosen Symbols")
def dlg_rm_imp_syms():
    st.markdown("### Are you sure you want to remove the imported symbols?")
    # Add "Yes" and "No" buttons to the dialog
    # Clicking "Yes" execute the function to remove the imported symbols
    st.button("Yes", key="yes_remove_imp_sym", on_click=remove_imported_symbols)
    # Clicking "No" closes the dialog
    st.button("No", key="no_remove_imp_sym")
    if st.session_state.yes_remove_imp_sym or st.session_state.no_remove_imp_sym:
        # Clear the dialog if the user clicks "Yes" or "No"
        st.rerun()


def remove_imported_symbols():
    logger.info('*** Removing Imported Symbols ***')
    # Get list of rows where Type is "Imported"
    imp_rows = st.session_state[ss_DfSym].index[st.session_state[ss_DfSym].Groups == 'Imported'].tolist()
    # Remove the imported symbols from the DataFrame
    st.session_state[ss_DfSym] = st.session_state[ss_DfSym].drop(imp_rows)
    # Remove the selected symbol from the sym_dpps_w and sym_dpps dictionaries
    if sym in st.session_state[ss_SymDpps_w]:
        del st.session_state[ss_SymDpps_w][sym]
    if sym in st.session_state[ss_SymDpps_d]:
        del st.session_state[ss_SymDpps_d][sym]

    st.session_state[ss_fDf_symbols_updated] = True
    st.session_state['imp_sums_removed'] = len(imp_rows)


# Create a function that returns the custom order index for sorting
# Define the custom order
custom_order = {'f': 0, '': 1, 'i': 2}


def fav_first_sort(value):
    # Return the custom order index, default to 3 if the value is not in custom_order
    return custom_order.get(value, 3)


def import_symbols(st, exp_sym):
    # Import Symbols from a csv file...
    sym_imp_cnt = 0
    # split the string into individual rows
    input_str = st.session_state.import_syms.getvalue().decode("utf-8")
    # csv.Dialect.skipinitialspace = True
    # rows_in = [row.strip().split(',') for row in input_str.split('\n')]
    # parsed_rows = []
    # for row in rows_in:
    #     new_row = []
    #     for col in row:
    #         new_row.append(col.replace('"', ''))
    #     parsed_rows.append(new_row)
    df_imported_syms = pd.read_csv(StringIO(input_str), header='infer')
    if len(df_imported_syms.columns) == import_cols_simple:
        df_imported_syms['Groups'] = 'Imported'
        df_imported_syms['Trend'] = ''
        df_imported_syms['WklyPrdStg'] = 0.0
        df_imported_syms['DlyPrdStg'] = 0.0
        df_imported_syms['wTop10Coint'] = ''
        df_imported_syms['wTop10Corr'] = ''
        df_imported_syms['wTop10xCorr'] = ''
        df_imported_syms['dTop10Corr'] = ''
        df_imported_syms['dTop10xCorr'] = ''

    # Drop the index column from the DataFrame
    df_imported_syms.reset_index(drop=True)
    # Add the symbol to the session_state DataFrame
    ss_df = st.session_state[ss_DfSym]
    sym_imp_cnt = 0
    # for row in parsed_rows:

    for i in range(df_imported_syms.shape[0]):
        row = df_imported_syms.iloc[i]
        if len(row) != import_cols_simple and len(row) != import_cols_full:
            logger.error(f"Skipping invalid row: {row}. Must have at least (2 or 10) columns.")
            continue
        # Add the symbol to the DataFrame
        if row['Symbol'] in ss_df.Symbol.values:
            df_imported_syms.drop(i)
            sym_imp_cnt += 1

    if df_imported_syms.shape[0] > 0:
        if len(df_imported_syms.columns) == import_cols_full:
            ss_df = df_imported_syms

    all_df_symbols = st.session_state[ss_AllDfSym]
    all_df_symbols = pd.concat([all_df_symbols, ss_df], ignore_index=True)
    st.session_state[ss_AllDfSym] = all_df_symbols
    st.session_state[ss_DfSym] = all_df_symbols

    if len(df_imported_syms) > 0:
        if sym_imp_cnt > 0:
            exp_sym.success(f'**[{sym_imp_cnt}] Symbols Successfully Imported.**', icon='👍')
            exp_sym.success(f'*** *Please detach the imported file.* ***')
        else:
            exp_sym.warning(f'**[!] No New Symbols Imported.**', icon='👎')
            exp_sym.warning(f'*** *Please detach the import file.* ***')


def add_new_symbols(st, exp_sym, syms):
    # Add the new symbols to the DataFrame
    all_df_symbols = st.session_state[ss_AllDfSym]
    df_symbols = st.session_state[ss_DfSym]
    added_symbols = []
    already_exists = []
    invalid_symbols = []
    for sym in syms:
        # Verify with yahoo finance that the symbol is valid, and get the long name
        logger.info(f"1. New Symbols to be added: {st.session_state.new_sym}")
        if sym == '':
            continue
        if sym not in st.session_state[ss_DfSym].Symbol.values:
            # Verify with yahoo finance that the symbol is valid, and get the long name
            new_ticker, long_name = getTickerLongName(sym)
            if long_name != '':
                new_row = pd.DataFrame({'Symbol': sym,
                                        'LongName': long_name,
                                        'Groups': 'Added', 'Trend': '',
                                        'DlyPrdStg': [0.0], 'WklyPrdStg': [0.0],
                                        'wTop10Coint': [''], 'wTop10xCoint': [''],
                                        'wTop10Corr': [''], 'wTop10xCorr': [''],
                                        'dTop10Corr': [''], 'dTop10xCorr': ['']})
                # Add the symbol to the display DataFrame
                df_symbols = pd.concat([df_symbols, new_row], axis=0,
                                       ignore_index=True)
                df_symbols = df_symbols.sort_values(by='Symbol')
                df_symbols.reindex()

                # Add the symbol to the all symbols DataFrame
                all_df_symbols = pd.concat([all_df_symbols, new_row], axis=0,
                                           ignore_index=True)
                all_df_symbols = all_df_symbols.sort_values(by='Symbol')
                all_df_symbols.reindex()

                added_symbols.append(sym)
            else:
                invalid_symbols.append(sym)
        else:
            already_exists.append(sym)

    df_symbols.reindex()
    all_df_symbols.reindex()
    st.session_state[ss_DfSym] = df_symbols
    st.session_state[ss_AllDfSym] = all_df_symbols

    if len(added_symbols) == 0:
        txt = ''
        if len(already_exists) > 0:
            txt += f"Symbols Already Exist: {already_exists}"
        txt += f"\n\nNo New Symbols to Add."
        if len(invalid_symbols) > 0:
            txt += f"\n\nInvalid Symbols: {invalid_symbols}"
        st.session_state['exp_sym'].warning(txt)
        return

    for sym in added_symbols:
        # Create a datetime that is 5 days ago
        five_days_ago = datetime.now() - timedelta(days=5)
        # Create a Daily PricePredict object for the symbol
        pp_d = PricePredict(ticker=sym, period=PricePredict.PeriodDaily,
                            model_dir=model_dir,
                            chart_dir=chart_dir,
                            preds_dir=preds_dir)
        pp_d.last_analysis = datetime.now()
        st.session_state[ss_SymDpps_d][sym] = pp_d
        # Create a Weekly PricePredict object for the symbol
        pp_w = PricePredict(ticker=sym,
                            period=PricePredict.PeriodWeekly,
                            model_dir=model_dir,
                            chart_dir=chart_dir,
                            preds_dir=preds_dir)
        pp_w.last_analysis = datetime.now()
        st.session_state[ss_SymDpps_w][sym] = pp_w

    txt = f"New Symbols Added: {added_symbols}"
    if len(already_exists) > 0:
        txt += f"\n\nSymbols Already Exist: {already_exists}"
    if len(invalid_symbols) > 0:
        txt += f"\n\nInvalid Symbols: {invalid_symbols}"
    st.session_state['exp_sym'].warning(txt)

    if len(added_symbols) > 0:
        # await analyze_symbols(prog_bar, st.session_state[ss_DfSym])
        with exp_sym, st.spinner("Analyzing Symbols (Please be patient)..."):
            # Run an analysis on all symbols
            prog_bar = exp_sym.progress(0, "Analyzing Symbols")
            analyze_symbols(st, prog_bar, df_symbols,
                            added_syms=added_symbols)

        st.info("Please Refresh Page After Adding Symbols.")
        streamlit_js_eval(js_expressions="parent.window.location.reload()")


def display_symbol_charts(interactive_charts=True):
    if (hasattr(st.session_state, ss_DfSym) is True and
        (hasattr(st.session_state, 'dfSymbols') and
         ((hasattr(st.session_state.dfSymbols, 'selection') is False)
            or (len(st.session_state.dfSymbols.selection.rows) == 0)))):
        return

    df_symbols = st.session_state[ss_DfSym]
    if hasattr(st.session_state[ss_DfSym], 'Symbol'):
        if hasattr(st.session_state, 'dfSymbols'):
            img_sym = st.session_state[ss_DfSym].Symbol[st.session_state.dfSymbols.selection.rows[0]]

    # Get todays date as a string in the format "YYYY-MM-DD"
    today = time.strftime("%Y-%m-%d")

    # Get current day of week
    day_of_week = datetime.today().weekday()

    img_cols = st.columns([1], gap='small', vertical_alignment='center')
    with (img_cols[0]):
        chart_missing = False
        # Ticker Chosen: Show tickers charts and data
        if 'img_sym' in locals() and img_sym is not None:
            # if ss_DfSym in st.session_state and 'Symbol' in st.session_state[ss_DfSym]:
            ss_df = st.session_state[ss_DfSym]
            sym_longName = ss_df.loc[ss_df['Symbol'] == img_sym, 'LongName'].iloc[0]
            # Within the main window...
            st.markdown(f"## - {img_sym}: {sym_longName} -")
            st.markdown(f"### Weekly Chart")
            sym_indx = img_sym + "_" + 'W'
            if sym_indx in st.session_state['sym_charts']:
                w_img_file = 'charts/' + st.session_state['sym_charts'][sym_indx]
            else:
                w_img_file = None
            if w_img_file is not None and img_sym not in w_img_file:
                w_img_file = get_sym_image_file(img_sym, 'W')
                if os.path.exists(w_img_file) is False:
                    del st.session_state['sym_charts'][sym_indx]
                    w_img_file = None
            if w_img_file is not None:
                # Get the date from the chart image file name
                wk_mtch = re.match(r'.*_(\d{4}-\d{2}-\d{2})\s+', w_img_file)
                if wk_mtch is not None:
                    wk_chart_date = wk_mtch.group(1)
                    wk_chart_dt = datetime.strptime(wk_chart_date, "%Y-%m-%d")
                else:
                    raise ValueError(f"Error parsing date from Weekly chart file name: {w_img_file}")
                # Get date of current week's friday
                todays_dt = datetime.today()
                todays_friday = todays_dt
                if todays_dt.weekday() != 4:
                    todays_friday = todays_dt - timedelta(days=todays_friday.weekday() - 4)
                if wk_chart_dt < todays_friday:
                    st.markdown("#### *** Chart May Not Be Current ***")
                # Display the daily chart image
                try:
                    if interactive_charts:
                        try:
                            pp_w = st.session_state[ss_SymDpps_w][img_sym]
                            file_path, fig = pp_w.gen_prediction_chart(save_plot=False, show_plot=True)
                            # st.plotly_chart(fig, use_container_width=True, width=800, html_width=800)
                            st.pyplot(fig, use_container_width=True)
                        except Exception as e:
                            st.image(w_img_file, use_container_width=True)
                            st.warning(f"Error displaying interactive chart: {e}")
                    else:
                        st.image(w_img_file, use_container_width=True)
                    st.button("Interactive Chart", key="b_interactive_chart")
                    if st.session_state.b_interactive_chart:
                        pp_w = st.session_state[ss_SymDpps_w][img_sym]
                        file_path, fig = pp_w.gen_prediction_chart(save_plot=False, show_plot=True)
                        fig.show()
                except Exception as e:
                    logger.error(f"Error displaying chart [{w_img_file}]:\n{e}")
                # Create expander for prediction analysis of weekly chart
                if img_sym in st.session_state[ss_SymDpps_w]:
                    pp = st.session_state[ss_SymDpps_w][img_sym]
                else:
                    return
                expd_WklyPa = st.expander("Weekly Chart Prediction Analysis...", expanded=False)
                expd_WklyPa.json(json.dumps(pp.analysis))
                indx = ss_df.index[ss_df['Symbol'] == img_sym]
                if ss_df.loc[indx]['WklyPrdStg'] is None:
                    val = None
                else:
                    val = ss_df.loc[indx]['WklyPrdStg'].iloc[0]
                st.markdown(f"**Weekly Prediction Strength**: {val}")
            else:
                st.markdown("#### *** No Weekly Chart Available ***")
                st.button("Generate Charts", key="gen_weekly_chart")
            st.markdown(f"### Daily Chart")
            sym_indx = img_sym + "_" + 'D'
            if sym_indx in st.session_state['sym_charts']:
                d_img_file = 'charts/' + st.session_state['sym_charts'][sym_indx]
                if os.path.exists(d_img_file) is False:
                    del st.session_state['sym_charts'][sym_indx]
                    d_img_file = None
            else:
                d_img_file = None
            if d_img_file is not None and img_sym not in d_img_file:
                d_img_file = get_sym_image_file(img_sym, 'D')
            if d_img_file is not None:
                # Get the date from the chart image file name
                dy_mtch = re.match(r'.*_(\d{4}-\d{2}-\d{2})\s+', w_img_file)
                if dy_mtch is not None:
                    dy_chart_date = dy_mtch.group(1)
                    dy_chart_dt = datetime.strptime(dy_chart_date, "%Y-%m-%d")
                else:
                    raise ValueError(f"Error parsing date from Daily chart file name: {w_img_file}")
                # Get date less 1 days, if a weekday
                # Get date less 2 days. if a weekend
                if dy_chart_dt.weekday() < 5:
                    chk_dt = datetime.today() - timedelta(days=1)
                else:
                    chk_dt = datetime.today() - timedelta(days=2)
                if dy_chart_dt < chk_dt:
                    st.markdown("#### *** Chart May Not Be Current ***")
                # Display the daily chart image
                try:
                    if interactive_charts:
                        try:
                            pp_d = st.session_state[ss_SymDpps_d][img_sym]
                            file_path, fig = pp_d.gen_prediction_chart(save_plot=False, show_plot=True)
                            # st.pyplot(fig, use_container_width='auto')
                            st.pyplot(fig, use_container_width=True)
                        except Exception as e:
                            st.image(d_img_file, use_container_width=True)
                            st.warning(f"Error displaying interactive chart: {e}")
                    else:
                        st.image(d_img_file, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error displaying chart [{d_img_file}]:\n{e}")
                # Create expander for prediction analysis of daily chart
                pp = st.session_state[ss_SymDpps_d][img_sym]
                expd_WklyPa = st.expander("Daily Chart Prediction Analysis...", expanded=False)
                expd_WklyPa.json(json.dumps(pp.analysis))
                indx = ss_df.index[ss_df['Symbol'] == img_sym]
                if ss_df.loc[indx]['DlyPrdStg'] is None:
                    val = None
                else:
                    val = ss_df.loc[indx]['DlyPrdStg'].iloc[0]
                st.markdown(f"**Daily Prediction Strength**: {val}")
            else:
                st.markdown("#### *** No Daily Chart Available ***")
                st.button("Generate Charts", key="gen_daily_chart")

            st.markdown(f"=====   =====   =====   =====   ====   ====   ====    ====   ====   =====   =====   =====   =====   =====   =====   =====   =====")

            expd_senti = st.expander("Sentiment...", expanded=False)
            sent_col1, sent_col2 = expd_senti.columns(2)
            sent_col1.button("Sentiment Analysis", key='sb_Sentiment')
            sent_col2.checkbox("Force Refresh", key='cb_ForceSentiment')
            if 'sb_Sentiment' in st.session_state and st.session_state.sb_Sentiment:
                if hasattr(pp, 'sentiment_text') is False or pp.sentiment_text == '' or st.session_state.cb_ForceSentiment:
                    pp.groq_sentiment()
                expd_senti.markdown(pp.sentiment_text + '\n\n' +
                                    '```json\n' + json.dumps(pp.sentiment_json, indent=3) + '\n```')

            if img_sym not in st.session_state[ss_SymDpps_d]:
                logger.error(f"Symbol [{img_sym}] not found in PricePredict objects")
            else:
                pp = st.session_state[ss_SymDpps_d][img_sym]
                expdr_corr = st.expander("**Correlations**", expanded=False)
                col1, col2, col3 = expdr_corr.columns(3)

                col1.markdown('**Top 10 Cointegrated**')
                df_to10coint = pd.DataFrame(data=pp.top10coint, columns=['Symbol', 'Coint PValue'])
                styl_to10coint = df_to10coint.style.set_properties(**{'boarder': '2px solid #ccc'})
                col1.table(styl_to10coint)

                col2.markdown('**Top 10 Correlated**')
                df_to10corr = pd.DataFrame(data=pp.top10corr, columns=['Symbol', 'Correlated'])
                styl_to10corr = df_to10corr.style.set_properties(**{'boarder': '2px solid #ccc'})
                col2.table(styl_to10corr)
                # col1.table(pp.top10corr)

                col3.markdown('**Top 10 X-Correlated**')
                df_to10xcorr = pd.DataFrame(data=pp.top10xcorr, columns=['Symbol', 'xCorrelated'])
                styl_to10xcorr = df_to10xcorr.style.set_properties(**{'boarder': '2px solid #ccc'})
                col3.table(styl_to10xcorr)
                # col2.table(pp.top10xcorr)

                if img_sym in st.session_state[ss_SymDpps_d]:
                    pp = st.session_state[ss_SymDpps_d][img_sym]
                    ticker_data = pp.ticker_data
                    st.markdown(f"=====   =====   =====   =====   ====   ====   ====    ====   ====   =====   =====   =====   =====   =====   =====   =====   =====")
                    if ticker_data is not None:
                        for key in ticker_data:
                            if key == 'address1':
                                expdr_biz = st.expander(f"**Symbol Basics**:")
                            elif key == 'longBusinessSummary':
                                expdr_biz = st.expander(f"**Company Info**:")
                            elif key == 'auditRisk':
                                expdr_biz = st.expander(f"**Investor Info**:")
                            elif key == 'maxAge':
                                expdr_biz = st.expander(f"**Index Data**:")
                            elif key == 'companyOfficers':
                                # expdr_biz.markdown(f"**{key}**:")
                                expdr_biz = st.expander(f"**{key}**:")
                                jsonStr = ticker_data[key]
                                for jstr in jsonStr:
                                    if jstr is not None:
                                        # Format the values of the jsonStr..,
                                        # expdr_biz.markdown(f"{json.dumps(jstr, indent=4)}")
                                        for key in jstr:
                                            biz_col1, biz_col2 = expdr_biz.columns(2)
                                            biz_col1.markdown(f"**{key}**:")
                                            biz_col2.markdown(f"{jstr[key]}")
                                        expdr_biz.markdown(f"### -------------------------------------")
                            elif key == 'maxAge':
                                for key_ in ticker_data:
                                    if key_ is not None:
                                        # Format the values of the jsonStr..,
                                        # expdr_biz.markdown(f"{json.dumps(jstr, indent=4)}")
                                        biz_col1, biz_col2 = expdr_biz.columns(2)
                                        biz_col1.markdown(f"**{key_}**:")
                                        biz_col2.markdown(f"{ticker_data[key_]}")
                                break
                            else:
                                expdr_biz.markdown(f"**{key}**: {ticker_data[key]}")
                    else:
                        st.markdown(f"**No Data Ticker Data Available**")

    if (('gen_weekly_chart' in st.session_state and st.session_state.gen_weekly_chart
         or ('gen_daily_chart' in st.session_state and st.session_state.gen_daily_chart))):
        pp_w = st.session_state[ss_SymDpps_w][img_sym]
        pp_d = st.session_state[ss_SymDpps_d][img_sym]
        pp_w.gen_prediction_chart(save_plot=True,  show_plot=False)
        pp_d.gen_prediction_chart(save_plot=True, show_plot=False)
        create_charts_dict(st)
        st.rerun()

    # As needed, save_plot out the updated DataFrames and PricePredict objects
    all_df_symbols = st.session_state[ss_AllDfSym]
    if len(all_df_symbols.columns) == import_cols_full:
        # Create a list of all symbols in the DataFrame all_df_symbols
        all_syms = all_df_symbols['Symbol'].tolist()
        # Sort the all_syms list
        all_syms.sort(key=fav_first_sort)
        # Place all symbols in all_syms into a single string.
        all_syms_str = ','.join(all_syms)
        # Create an md5 hash of the all_syms_str string
        all_syms_md5 = hashlib.md5(all_syms_str.encode()).hexdigest()
        if 'all_syms_md5' not in st.session_state:
            st.session_state['all_syms_md5'] = all_syms_md5
        if all_syms_md5 != st.session_state['all_syms_md5']:
            st.session_state['all_syms_md5'] = all_syms_md5
            logger.debug(f"5> Saving all_df_symbols to {guiAllSymbolsCsv}")
            all_df_symbols.to_csv(guiAllSymbolsCsv, index=False)
            logger.info("DataFrames and PricePredict objects saved")
    else:
        logger.error(f"Error all_df_symbols has too many columns [{len(all_df_symbols)}]: {all_df_symbols.columns}")


def get_sym_image_file(sym, period, path):
    today = time.strftime("%Y-%m-%d")
    img_file = f"{path}/{sym}_{period}_{today}.png"
    if os.path.isfile(img_file):
        return img_file
    else:
        # Find the most recent image file for the symbol
        files = os.listdir(path)
        # Isolate files that contain the symbol and period
        files = [file for file in files if sym in file and period in file]
        files.sort(reverse=True)
        for file in files:
            if sym in file and period in file:
                return f"{path}/{file}"
    return None


# ======================================================================
# Analyze Symbols ======================================================
# Analyze the symbols in the DataFrame, in a separate thread.
# ======================================================================
# async def analyze_symbols(prog_bar, df_symbols):
def analyze_symbols(st, prog_bar, df_symbols,
                    added_syms=None,
                    force_training=False,
                    force_analysis=False):
    logger.info("*** Analyzing Symbols: Started ***")
    all_df_symbols = st.session_state[ss_AllDfSym]
    total_syms = all_df_symbols.shape[0]
    i = 0
    # futures = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor = fp.ThreadPoolExecutor(max_workers=8)
    with fp.TaskManager(executor) as tm:
        logger.debug(f"=== ThreadPoolExecutor: Analyzing Symbols: Total Symbols: {total_syms}")
        # Loop through the symbols in the DataFrame
        for row in all_df_symbols.itertuples():

            if added_syms is not None and row.Symbol not in added_syms:
                # Skip all symbols not in the imported symbols list
                continue
            else:
                logger.info(f"imported_syms/added_sym: processing symbol [{row.Symbol}]")
            # Check if the model has a weekly PricePredict object
            if row.Symbol not in st.session_state[ss_SymDpps_w]:
                # Create a weekly PricePredict object for the symbol
                ppw = PricePredict(ticker=row.Symbol, period=PricePredict.PeriodWeekly,
                                   model_dir=model_dir,
                                   chart_dir=chart_dir,
                                   preds_dir=preds_dir)
                st.session_state[ss_SymDpps_w][row.Symbol] = ppw
                st.session_state[ss_SymDpps_w][row.Groups] = 'Added'
            else:
                ppw = st.session_state[ss_SymDpps_w][row.Symbol]

            if st.session_state[ss_forceTraining]:
                ppw.orig_data = None
            if (ppw.orig_data is None
                or row.Trend == ''
                or row.WklyPrdStg == 0.0):
                # Get the datetime of 5 days ago
                five_days_ago = datetime.now() - timedelta(days=7)
                # This will force an update of the ppw object
                ppw.last_analysis = datetime.now()
                ppw.force_training = True

            logger.info(f"Weekly - Pull data for model: {row.Symbol}")
            try:
                task_pull_data(row.Symbol, ppw)
                logger.info(f"Weekly - Train and Predict: {row.Symbol}")
                # future = executor.submit(task_train_predict_report, row.Symbol, ppw)
                # futures.append(future)
                tm.submit(task_train_predict_report, row.Symbol, ppw,
                          added_syms=added_syms,
                          force_training=force_training,
                          force_analysis=force_analysis)
                total_syms += 1
            except Exception as e:
                logger.error(f"Error processing symbol: {row.Symbol}\n{e}")

            # Check if the model has a daily PricePredict object
            if row.Symbol not in st.session_state[ss_SymDpps_d]:
                # Create a daily PricePredict object for the symbol
                ppd = PricePredict(ticker=row.Symbol, period=PricePredict.PeriodDaily,
                                   model_dir=model_dir,
                                   chart_dir=chart_dir,
                                   preds_dir=preds_dir)
                st.session_state[ss_SymDpps_d][row.Symbol] = ppd
            else:
                ppd = st.session_state[ss_SymDpps_d][row.Symbol]

            if st.session_state[ss_forceTraining]:
                ppd.orig_data = None
            if (ppd.orig_data is None
                or row.Trend == ''
                or row.WklyPrdStg == 0.0):
                # Get the datetime of 5 days ago
                five_days_ago = datetime.now() - timedelta(days=3)
                # This will force an update of the ppd object
                ppd.last_analysis = datetime.now()
                ppd.force_training = True

            logger.info(f"Daily - Pull data for model: {row.Symbol}")
            try:
                task_pull_data(row.Symbol, ppd)
                logger.info(f"Daily - Train and Predict: {row.Symbol}")
                # future = executor.submit(task_train_predict_report, row.Symbol, ppd)
                # futures.append(future)
                tm.submit(task_train_predict_report, row.Symbol, ppd,
                          added_syms=added_syms,
                          force_training=force_training,
                          force_analysis=force_analysis)
                total_syms += 1
            except Exception as e:
                logger.error(f"Error pull-train-predict on symbol: {row.Symbol}\n{e}")

            i += 1
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Analyzing: {row.Symbol} ({i}/{total_syms})")

        logger.debug(f"=== Checking on completed futures ...")
        # for future in concurrent.futures.as_completed(futures):
        sym_ = ''
        for future in tm.as_completed():
            logger.debug(f"=== Checking futures ...")

            i += 1
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Analysis Completed: {sym_} ({i}/{total_syms})")

            if isinstance(future, Exception):
                logger.error(f"Error processing symbol: {future.args[0], future.args[1]} {future.result}")
                continue
            else:
                sym_, pp_ = future.result
                if sym_ is not None and pp_.period == PricePredict.PeriodWeekly:
                    st.session_state[ss_SymDpps_w][sym_] = pp_
                elif sym_ is not None and pp_.period == PricePredict.PeriodDaily:
                    st.session_state[ss_SymDpps_d][sym_] = pp_
                else:
                    logger.error(f"Error PricePredict Object has invalid period value: {pp_.period}")

    sym_correlations('Weekly', st, st.session_state[ss_SymDpps_w], prog_bar)
    sym_correlations('Daily', st, st.session_state[ss_SymDpps_d], prog_bar)

    all_df_symbols = st.session_state[ss_AllDfSym]

    st.session_state[ss_DfSym] = df_symbols

    st.session_state[ss_DfSym] = update_viz_data(st, all_df_symbols)

    logger.debug(f"6> Saving all_df_symbols to {guiAllSymbolsCsv}")
    all_df_symbols = st.session_state[ss_AllDfSym]
    df_symbols = st.session_state[ss_DfSym]
    with st.spinner("Saving PricePredict Objects..."):
        merge_and_save(all_df_symbols, df_symbols)

    # Save out the updated DataFrames and PricePredict objects
    store_pp_objects(st, prog_bar)
    logger.info("--- Analyzing Symbols: Completed ---")

    with st.spinner("Cleaning up..."):
        # Remove old PricePredict objects
        del_d_models = model_cleanup(PricePredict.PeriodDaily, st.session_state[ss_AllDfSym])
        del_w_models = model_cleanup(PricePredict.PeriodWeekly, st.session_state[ss_AllDfSym])
        del_1h_models = model_cleanup(PricePredict.Period1hour, st.session_state[ss_AllDfSym])
        del_5min_models = model_cleanup(PricePredict.Period5min, st.session_state[ss_AllDfSym])
        del_1min_models = model_cleanup(PricePredict.Period1min, st.session_state[ss_AllDfSym])
        print(f"Deleted {del_d_models} Daily PricePredict objects")
        print(f"Deleted {del_w_models} Weekly PricePredict objects")
        print(f"Deleted {del_1h_models} 1Hour PricePredict objects")
        print(f"Deleted {del_5min_models} 5Min PricePredict objects")
        print(f"Deleted {del_1min_models} 1Min PricePredict objects")

        del_d_ppo = ppo_cleanup(PricePredict.PeriodDaily, st.session_state[ss_AllDfSym])
        del_w_ppo = ppo_cleanup(PricePredict.PeriodWeekly, st.session_state[ss_AllDfSym])
        del_1h_ppo = ppo_cleanup(PricePredict.Period1hour, st.session_state[ss_AllDfSym])
        del_5min_ppo = ppo_cleanup(PricePredict.Period5min, st.session_state[ss_AllDfSym])
        del_1min_ppo = ppo_cleanup(PricePredict.Period1min, st.session_state[ss_AllDfSym])
        print(f"Deleted {del_d_ppo} Daily PricePredict objects")
        print(f"Deleted {del_w_ppo} Weekly PricePredict objects")
        print(f"Deleted {del_1h_ppo} 1Hour PricePredict objects")
        print(f"Deleted {del_5min_ppo} 5Min PricePredict objects")
        print(f"Deleted {del_1min_ppo} 1Min PricePredict objects")

        del_d_charts = charts_cleanup(st, PricePredict.PeriodDaily)
        del_w_charts = charts_cleanup(st, PricePredict.PeriodWeekly)
        del_1h_charts = charts_cleanup(st, PricePredict.Period1hour)
        del_5min_charts = charts_cleanup(st, PricePredict.Period5min)
        del_1min_charts = charts_cleanup(st, PricePredict.Period1min)
        print(f"Deleted {del_d_charts} Daily Chart images")
        print(f"Deleted {del_w_charts} Weekly Chart images")
        print(f"Deleted {del_1h_charts} 1Hour Chart images")
        print(f"Deleted {del_5min_charts} 5Min Chart images")
        print(f"Deleted {del_1min_charts} 1Min Chart images")


def load_pp_objects__(st):
    sym_dpps_d_ = {}
    sym_dpps_w_ = {}

    min_dil_size = 70000

    # Check if the PricePredict objects files exist
    if not os.path.exists(dill_sym_dpps_d) or not os.path.exists(dill_sym_dpps_w):
        logger.error("PricePredict object files do not exist")
        st.session_state[ss_SymDpps_d] = sym_dpps_d_
        st.session_state[ss_SymDpps_w] = sym_dpps_w_
        sync_dpps_objects(st, None)
        return sym_dpps_d_, sym_dpps_w_

    # Make sure that files are not zero length
    if (os.path.getsize(dill_sym_dpps_d) > min_dil_size
            and os.path.getsize(dill_sym_dpps_w) > min_dil_size):
        # Load PricePredict objects
        logger.info("Loading PricePredict objects")
        try:
            with open(dill_sym_dpps_d, "rb") as f:
                sym_dpps_d_ = dill.load(f)
            with open(dill_sym_dpps_w, "rb") as f:
                sym_dpps_w_ = dill.load(f)
            # Create Backups files after a successful load...
            # Copy the PricePredict objects to the backup files
            logger.info("Copying PricePredict objects to backup files")
            shutil.copy(dill_sym_dpps_d, dillbk_sym_dpps_d)
            shutil.copy(dill_sym_dpps_w, dillbk_sym_dpps_w)

        except Exception as e:
            logger.error(f"Error loading PricePredict objects via dill: {e}")
    else:
        logger.error("Error a PricePredict object is too small")
        logger.error("Restoring picked PricePredict objects from backup")
        if (os.path.getsize(dillbk_sym_dpps_d) > min_dil_size
                and os.path.getsize(dillbk_sym_dpps_w) > min_dil_size):
            try:
                with open(dillbk_sym_dpps_d, "rb") as f:
                    sym_dpps_d_ = dill.load(f)
                with open(dillbk_sym_dpps_w, "rb") as f:
                    sym_dpps_w_ = dill.load(f)
            except Exception as e:
                logger.error(f"Error loading PricePredict backup objects via dill: {e}")
        else:
            logger.error("Error the backup PricePredict object file is too small")
            logger.error("PricePredict objects were not restored")

    if sym_dpps_d_ is None:
        # Add a dummy PricePredict object so that other logic doesn't break
        sym_dpps_d_['___'] = PricePredict(ticker='___', period=PricePredict.PeriodDaily,
                                          model_dir=model_dir,
                                          chart_dir=chart_dir,
                                          preds_dir=preds_dir)
    if sym_dpps_w_ is None:
        # Add a dummy PricePredict object so that other logic doesn't break
        sym_dpps_w_['___'] = PricePredict(ticker='___', period=PricePredict.PeriodWeekly,
                                          model_dir=model_dir,
                                          chart_dir=chart_dir,
                                          preds_dir=preds_dir)

    st.session_state[ss_SymDpps_d] = sym_dpps_d_
    st.session_state[ss_SymDpps_w] = sym_dpps_w_

    sync_dpps_objects(st, None)

    return sym_dpps_d_, sym_dpps_w_

def load_pp_objects(st):

    logger.debug("Loading PricePredict objects..,")

    sym_dpps_d_ = {}
    sym_dpps_w_ = {}

    min_dil_size = 70000
    ppo_dir = './ppo/'

    prog_bar = st.progress(0, "Finding PricePredict objects...")
    # Find the latest .dill files in the ./ppo directory for any given symbol.
    dill_files = {}
    with os.scandir(ppo_dir) as entries:
        tot_entries = len(list(os.scandir(ppo_dir)))
        i = 0
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.dill'):
                sym, period = entry.name.split('_')[:2]
                sym_period = sym + '_' + period
                # Find a key in dill_files that starts with sym and place the entry
                # the variable curr_entry.
                curr_entry = next((v for k, v in dill_files.items() if k.startswith(sym_period)), None)
                if curr_entry is not None:
                    if entry.name > curr_entry.name:
                        # Replace the entry with the newer file
                        dill_files[sym_period] = entry
                else:
                    dill_files[sym_period] = entry
            i += 1
            prog_bar.progress(int(i / tot_entries) * 100, f"Finding PricePredict objects: {entry.name} ({i}/{tot_entries})")

    prog_bar.progress(0, "Loading PricePredict objects...")
    tot_entries = len(dill_files.keys())
    i = 0
    for sym_period in dill_files.keys():
        entry = dill_files[sym_period]
        if entry.is_file():
            sym = entry.name.split('_')[0]
            period = entry.name.split('_')[1]
            if period == 'W':
                try:
                    with open(entry, "rb") as f:
                        sym_dpps_w_[sym] = dill.load(f)
                except Exception as e:
                    logger.warning(f"Error loading PricePredict object [{sym}]: {e}")
            elif period == 'D':
                try:
                    with open(entry, "rb") as f:
                        sym_dpps_d_[sym] = dill.load(f)
                except Exception as e:
                    logger.warning(f"Error loading PricePredict object [{sym}]: {e}")
            i += 1
            prog_bar.progress(int(i / tot_entries) * 100, f"Loading PricePredict objects: {entry.name} ({i}/{tot_entries})")

    st.session_state[ss_SymDpps_d] = sym_dpps_d_
    st.session_state[ss_SymDpps_w] = sym_dpps_w_

    sync_dpps_objects(st, prog_bar)
    prog_bar.empty()

    return sym_dpps_d_, sym_dpps_w_

def sync_dpps_objects(st, prog_bar):
    logger.info("Remove PricePredict objects that are not in the DataFrame")
    df_symbols = st.session_state[ss_AllDfSym]
    if ss_SymDpps_d not in st.session_state.keys() or ss_SymDpps_w not in st.session_state.keys():
        return
    if st.session_state[ss_SymDpps_d] is None or st.session_state[ss_SymDpps_w] is None:
        return
    sym_dpp_d = st.session_state[ss_SymDpps_d].copy()
    sym_dpp_w = st.session_state[ss_SymDpps_w].copy()
    object_removed = []
    i = 0
    total_syms = len(sym_dpp_d)
    for sym in sym_dpp_d:
        if sym not in df_symbols.Symbol.values:
            del st.session_state[ss_SymDpps_d][sym]
        # Verify that dpp_d object is a PricePredict object
        dpp_d = sym_dpp_d[sym]
        if isinstance(dpp_d, PricePredict):
            # Test pickling the object to a buffer.
            buffer = io.BytesIO()
            try:
                dill.dump(dpp_d, buffer)
            except Exception as e:
                logger.warning(f"WARN: pickling PricePredict dpp_d object [{sym}]: {e}")
                bo = dill.detect.badobjects(dpp_d, depth=2)
                logger.error(f"Bad objects in [{dpp_d.ticker}]:w: {bo}")
                objgraph.show_refs(dpp_d, filename=f'objgraph_{dpp_d.ticker}_dpp_d.png')
                # Remove the unpicklable object
                del st.session_state[ss_SymDpps_d][sym]
                object_removed.append(sym+':d')
        i += 1
        if prog_bar is not None:
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Validating Daily Objects: {sym} ({i}/{total_syms})")

    i = 0
    total_syms = len(sym_dpp_d)
    for sym in sym_dpp_w:
        if sym not in df_symbols.Symbol.values:
            del st.session_state[ss_SymDpps_w][sym]
        # Verify that dpp_d object is a PricePredict object
        dpp_w = sym_dpp_w[sym]
        if isinstance(dpp_w, PricePredict):
            # Test pickling the object to a buffer.
            buffer = io.BytesIO()
            try:
                dill.dump(dpp_w, buffer)
            except Exception as e:
                logger.warning(f"WARN: pickling PricePredict dpp_w object [{sym}]: {e}")
                bo = dill.detect.badobjects(dpp_w, depth=2)
                logger.error(f"Bad objects in [{dpp_w.ticker}]:w: {bo}")
                objgraph.show_refs(dpp_d, filename=f'objgraph_{dpp_d.ticker}_dpp_w.png')
                # Remove the unpicklable object
                del st.session_state[ss_SymDpps_w][sym]
                object_removed.append(sym+':w')
        i += 1
        if prog_bar is not None:
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Validating Weekly Objects: {sym} ({i}/{total_syms})")

    if len(object_removed) > 0:
        logger.info(f'Deleted {len(object_removed)} Daily PricePredict objects: [{','.join(object_removed)}]')
        st.session_state['exp_sym'].warning(f"Removed UnPicklable PricePredict objects: {object_removed}")

    # Make sure that we have PricePredict objects for all the symbols in the DataFrame
    for sym in df_symbols.Symbol.values:
        if sym not in st.session_state[ss_SymDpps_d]:
            # Create missing PricePredict objects
            pp = PricePredict(ticker=sym, period=PricePredict.PeriodDaily, logger=logger,
                              model_dir=model_dir,
                              chart_dir=chart_dir,
                              preds_dir=preds_dir)
            st.session_state[ss_SymDpps_d][sym] = pp
        if sym not in st.session_state[ss_SymDpps_w]:
            # Create missing PricePredict objects
            pp = PricePredict(ticker=sym, period=PricePredict.PeriodWeekly, logger=logger,
                              model_dir=model_dir,
                              chart_dir=chart_dir,
                              preds_dir=preds_dir)
            st.session_state[ss_SymDpps_w][sym] = pp


def store_pp_objects__(st, prog_bar):

    sync_dpps_objects(st, prog_bar)
    logger.info("Saving PricePredict objects (Daily Object)...")
    st.session_state['exp_sym'].info("Saving PricePredict Daily objects...")
    # Save out the PricePredict objects
    sym_dpps_d_ = st.session_state[ss_SymDpps_d]
    try:
        if len(sym_dpps_d_) > 0:
            with open(dill_sym_dpps_d, "wb") as f:
                dill.dump(sym_dpps_d_, f)
        else:
            # Truncate the pickle file
            with open(dill_sym_dpps_d, "wb") as f:
                f.truncate()
    except Exception as e:
        logger.error(f"Error {dill_sym_dpps_d} - len[{len(sym_dpps_d_)}]: {e}")
        bo = dill.detect.badobjects(sym_dpps_d_, depth=2)
        logger.error(f"Bad objects in [{sym_dpps_d_.ticker}]:w: {bo}")
        objgraph.show_refs(sym_dpps_d_, filename=f'objgraph_{sym_dpps_d_.ticker}_dpp_d.png')
    st.session_state['exp_sym'].error(f"Error saving Daily PricePredict objects: {e}")

    logger.info("Saving PricePredict objects (Weekly Object)...")
    st.session_state['exp_sym'].info("Saving PricePredict Weekly objects...")
    sym_dpps_w_ = st.session_state[ss_SymDpps_w]
    try:
        if len(sym_dpps_w_) > 0:
            with open(dill_sym_dpps_w, "wb") as f:
                dill.dump(sym_dpps_w_, f)
        else:
            # Truncate the pickle file
            with open(dill_sym_dpps_w, "wb") as f:
                f.truncate()

    except Exception as e:
        logger.error(f"Error saving  {dill_sym_dpps_w} - len[{len(sym_dpps_w_)}]: {e}")
        bo = dill.detect.badobjects(sym_dpps_w_, depth=2)
        logger.error(f"Bad objects in [{sym_dpps_w_.ticker}]:w: {bo}")
        objgraph.show_refs(sym_dpps_w_, filename=f'objgraph_{sym_dpps_w_.ticker}_dpp_d.png')
        st.session_state['exp_sym'].error(f"Error saving Weekly PricePredict objects: {e}")


def store_pp_objects(st, prog_bar):

    sync_dpps_objects(st, prog_bar)

    logger.info("Saving PricePredict objects (Weekly Object)...")

    failed_ppws = []
    for sym_w in st.session_state[ss_SymDpps_w]:
        ppw = st.session_state[ss_SymDpps_w][sym_w]
        if ppw is not None:
            ticker = ppw.ticker
            if ppw.date_data is None:
                continue
            last_date = ppw.date_data.iloc[-1].strftime("%Y-%m-%d")
            period = ppw.period
            obj_file_name = f"{ticker}_{period}_{last_date}.dill"
            file_path = './ppo/' + obj_file_name
            try:
                with open(file_path, "wb") as f:
                    dill.dump(ppw, f)
            except Exception as e:
                bo = dill.detect.badobjects(ppw, depth=2)
                logger.error(f"Bad objects in [{ppw.ticker}]:w: {bo}")
                objgraph.show_refs(sym_dpps_w_, filename=f'objgraph_{ppw.ticker}_dpp_d.png')
                logger.error(f"Error saving PricePredict object [{sym_w}]: {e}")
                failed_ppws.append(sym_w)
    if len(failed_ppws) > 0:
        st.warning(f"Failed to save PricePredict objects: {failed_ppws}")

    logger.info("Saving PricePredict objects (Daily Object)...")

    failed_ppws = []
    for sym_d in st.session_state[ss_SymDpps_d]:
        ppd = st.session_state[ss_SymDpps_d][sym_d]
        if ppd is not None:
            ticker = ppd.ticker
            if ppd.date_data is None:
                continue
            last_date = ppd.date_data.iloc[-1].strftime("%Y-%m-%d")
            period = ppd.period
            obj_file_name = f"{ticker}_{period}_{last_date}.dill"
            file_path = './ppo/' + obj_file_name
            try:
                with open(file_path, "wb") as f:
                    dill.dump(ppd, f)
            except Exception as e:
                bo = dill.detect.badobjects(ppd, depth=2)
                logger.error(f"Bad objects in [{ppd.ticker}]:w: {bo}")
                objgraph.show_refs(sym_dpps_w_, filename=f'objgraph_{ppd.ticker}_dpp_d.png')
                logger.error(f"Error saving PricePredict object [{sym_d}]: {e}")
                failed_ppws.append(sym_d)
    if len(failed_ppws) > 0:
        st.warning(f"Failed to save PricePredict objects: {failed_ppws}")

def task_pull_data(symbol_, dpp):
    # Get datetime 24 hours ago
    ago24hrs = datetime.now() - timedelta(days=1)

    logger.info(f"Pulling data for {symbol_}...")
    if dpp.last_analysis is not None:
        if dpp.last_analysis > ago24hrs and dpp.model is not None:
            logger.info(f"PricePredict object is already up-to-date: {symbol_}")
            return symbol_, dpp

    # Set the end date to today...
    end_date = datetime.now().strftime("%Y-%m-%d")
    # Set the start date to 4 years before the end_date...
    start_date = (datetime.strptime(end_date, "%Y-%m-%d")
                  - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    # Pull the data and cache it...
    dpp.cache_training_data(symbol_, start_date, end_date, dpp.period)
    if len(dpp.orig_data) < 150:
        logger.error(f"Error: Not enough data for [{symbol_}] [{len(dpp.orig_data) }], for training. So, won't train or predict.")
        # TODO: Add a problem property to the PricePredict object that can be displayed for
        #       objects that have issues.
        return symbol_, dpp

    # Change the start date to 14 days before the end date...
    if dpp.period == PricePredict.PeriodWeekly:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d")
                      - timedelta(days=600)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d")
                      - timedelta(days=365)).strftime("%Y-%m-%d")
    # Simply pulls and caches the prediction data...
    dpp.cache_prediction_data(symbol_, start_date, end_date, dpp.period)
    logger.info(f"Completed pulling data for {symbol_}...")
    return symbol_, dpp


def task_train_predict_report(symbol_, dpp, added_syms=None,
                              force_training=False, force_analysis=False):
    # Get datetime 24 hours ago
    ago24hrs = datetime.now() - timedelta(days=1)

    if dpp.last_analysis is not None:
        if dpp.last_analysis > ago24hrs and force_analysis is False and dpp.model is not None:
            logger.info(f"PricePredict object is already uptodate: {symbol_}")
            return symbol_, dpp
    # Process the cached data as needed...
    # - Trains and Saves a new model if needed
    # - Performs the prediction on the cached prediction data
    # - Generates the required charts and database updates
    if added_syms is None:
        added_syms = []
    if dpp.ticker in added_syms:
        force_training = True
    if force_training or dpp.last_analysis is None or dpp.model is None:
        logger.info(f"Training and predicting for {symbol_}...")
        dd = dpp.cached_train_data.dates_data
        dpp.cached_train_predict_report(force_training=True)
        logger.info(f"Completed training and predicting for {symbol_}...")
    else:
        logger.info(f"Predicting for {symbol_}...")
        dpp.cached_predict_report()
        logger.info(f"Completed predicting for {symbol_}...")

    return symbol_, dpp


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def create_charts_dict(st):
    # Create a dicts of the chart image files in ./charts
    # The format of the chart file name is <Symbol>_period_<date>.png
    logger.debug("Creating charts dictionary...")
    charts_dict = {}
    if os.path.exists('charts'):
        files = os.listdir('charts')
        files.sort(reverse=True)
        for file in files:
            if file.endswith('.png'):
                sym = file.split('_')[0]
                period = file.split('_')[1]
                index = sym + '_' + period
                if index not in charts_dict.keys():
                    charts_dict[sym + '_' + period] = file
    if len(charts_dict) > 0:
        st.session_state['sym_charts'] = charts_dict
    else:
        st.session_state['sym_charts'] = {}

def update_viz_data(st, all_df_symbols) -> pd.DataFrame:

    create_charts_dict(st)

    min_data_points = 50

    if ss_SymDpps_d in st.session_state.keys():
        sym_dpps_d = st.session_state[ss_SymDpps_d]
        all_df_symbols = st.session_state[ss_AllDfSym]
        # Reindex all_df_symbols on the Symbol column

        for sym in all_df_symbols.Symbol.values:
            if sym in sym_dpps_d:
                pp = sym_dpps_d[sym]
            else:
                pp = PricePredict(ticker=sym, period=PricePredict.PeriodDaily, logger=logger,
                                  model_dir=model_dir,
                                  chart_dir=chart_dir,
                                  preds_dir=preds_dir)
                sym_dpps_d[sym] = pp

            if pp.pred_strength is None:
                logger.warning(f"Symbol [{sym}] Period [{pp.period}] has no prediction strength value.")

            if sym not in all_df_symbols.Symbol.values:
                indx = -1
            else:
                indx = all_df_symbols.index[all_df_symbols.Symbol == sym]

            if pp.ticker_data is not None:
                all_df_symbols.loc[indx, 'Symbol'] = sym
                lname = '- No Data -'
                if 'longName' in pp.ticker_data:
                    lname = pp.ticker_data['longName']
                elif 'shortName' in pp.ticker_data:
                    lname = pp.ticker_data['shortName']
                all_df_symbols.loc[indx, 'LongName'] = lname
            d_trend = 'D:'
            if pp.pred_strength is not None:
                if abs(pp.pred_strength) < 0.5:
                    d_trend += 'f'
                elif pp.pred_strength < 0:
                    d_trend += 'd'
                else:
                    d_trend += 'u'
            else:
                d_trend += '_'

            all_df_symbols.loc[indx, 'Trend'] = d_trend
            prd_strg = pp.pred_strength
            if prd_strg is None:
                prd_strg = 0
            all_df_symbols.loc[indx, 'DlyPrdStg'] = f'{prd_strg:>,.4f}'
            all_df_symbols.loc[indx, 'dTop10Coint'] = json.dumps(pp.top10coint)
            all_df_symbols.loc[indx, 'dTop10Corr'] = json.dumps(pp.top10corr)
            all_df_symbols.loc[indx, 'dTop10xCorr'] = json.dumps(pp.top10xcorr)

    if ss_SymDpps_w in st.session_state.keys():
        sym_dpps_w = st.session_state[ss_SymDpps_w]
        for sym in all_df_symbols.Symbol.values:
            if sym in sym_dpps_w:
                pp = sym_dpps_w[sym]
            else:
                pp = PricePredict(ticker=sym, period=PricePredict.PeriodWeekly, logger=logger,
                                  model_dir=model_dir,
                                  chart_dir=chart_dir,
                                  preds_dir=preds_dir)
                sym_dpps_w[sym] = pp

            if pp.pred_strength is None:
                logger.warning(f"Symbol [{sym}] Period [{pp.period}] has no prediction strength value.")

            if sym not in all_df_symbols.Symbol.values:
                indx = -1
            else:
                indx = all_df_symbols.index[all_df_symbols.Symbol == sym]

            if pp.ticker_data is not None:
                all_df_symbols.loc[indx, 'Symbol'] = sym
                lname = '- No Data -'
                if 'longName' in pp.ticker_data:
                    lname = pp.ticker_data['longName']
                elif 'shortName' in pp.ticker_data:
                    lname = pp.ticker_data['shortName']
                all_df_symbols.loc[indx, 'LongName'] = lname
            w_trend = 'W:'
            if pp.pred_strength is not None:
                if abs(pp.pred_strength) < 0.5:
                    w_trend += 'f'
                elif pp.pred_strength < 0:
                    w_trend += 'd'
                else:
                    w_trend += 'u'
            else:
                w_trend += '_'

            all_df_symbols.loc[indx, 'Trend'] = w_trend + ' - ' + all_df_symbols.loc[indx, 'Trend'].values[0]
            prd_strg = pp.pred_strength
            if prd_strg is None:
                prd_strg = 0
            all_df_symbols.loc[indx, 'WklyPrdStg'] = f'{prd_strg:>,.4f}'
            all_df_symbols.loc[indx, 'wTop10Coint'] = json.dumps(pp.top10coint)
            all_df_symbols.loc[indx, 'wTop10Corr'] = json.dumps(pp.top10corr)
            all_df_symbols.loc[indx, 'wTop10xCorr'] = json.dumps(pp.top10xcorr)

    st.session_state[ss_AllDfSym] = all_df_symbols
    df_symbols = st.session_state[ss_DfSym]
    # Update the ss_DfSym DataFrame with the latest from all_df_symbols
    for row in df_symbols.itertuples():
        if row.Symbol in all_df_symbols.Symbol.values:
            indx1 = df_symbols.index[df_symbols.Symbol == row.Symbol]
            indx2 = all_df_symbols.index[all_df_symbols.Symbol == row.Symbol]
            df_symbols.loc[indx1, 'LongName'] = all_df_symbols.loc[indx2, 'LongName']
            df_symbols.loc[indx1, 'Groups'] = all_df_symbols.loc[indx2, 'Groups']
            df_symbols.loc[indx1, 'Trend'] = all_df_symbols.loc[indx2, 'Trend']
            df_symbols.loc[indx1, 'DlyPrdStg'] = all_df_symbols.loc[indx2, 'DlyPrdStg']
            df_symbols.loc[indx1, 'WklyPrdStg'] = all_df_symbols.loc[indx2, 'WklyPrdStg']
            df_symbols.loc[indx1, 'dTop10Coint'] = all_df_symbols.loc[indx2, 'dTop10Coint']
            df_symbols.loc[indx1, 'dTop10Corr'] = all_df_symbols.loc[indx2, 'dTop10Corr']
            df_symbols.loc[indx1, 'dTop10xCorr'] = all_df_symbols.loc[indx2, 'dTop10xCorr']
            df_symbols.loc[indx1, 'wTop10Coint'] = all_df_symbols.loc[indx2, 'wTop10Coint']
            df_symbols.loc[indx1, 'wTop10Corr'] = all_df_symbols.loc[indx2, 'wTop10Corr']
            df_symbols.loc[indx1, 'wTop10xCorr'] = all_df_symbols.loc[indx2, 'wTop10xCorr']

    st.session_state[ss_DfSym] = df_symbols

    return df_symbols


def sym_correlations(prd, st, sym_dpps, prog_bar):
    """
    This procedure performs correlation analysis for each PricePredict object
    against all other PricePredict objects amd update the PricePredict object's
    top10corr, top10xcorr, and top10coint properties.
    """
    logger.debug(f"Calculating Period [{prd}] Correlations...")

    # Minimum number of data points required for correlation calculations
    min_data_points = 200
    # Data points to request from Yahoo if data is needed
    req_data_points = 400

    # Correlations on the Daily Objects...
    all_df_symbols = st.session_state[ss_AllDfSym]
    sym_corr = {}
    i = 0
    item_cnt = len(sym_dpps)
    for tsym in sym_dpps:
        if tsym not in all_df_symbols.Symbol.values:
            # Skip symbols not in the DataFrame
            continue
        i += 1
        # Update the progress bar
        prog_bar.progress(int(i / item_cnt * 100), f"{prd} Correlations (1): {tsym} ({i}/{item_cnt})")

        target_sym = sym_dpps[tsym]
        target_sym.ticker = tsym  # Make sure the ticker is set correctly

        days_back = req_data_points
        if target_sym.period == PricePredict.PeriodWeekly:
            days_back = req_data_points * 7

        if target_sym.orig_data is None or len(target_sym.orig_data) < min_data_points:
            # Load up the data for the target symbol if it does not have enough data points
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            target_sym.fetch_data_yahoo(target_sym.ticker, start_date, end_date, target_sym.period)

        if target_sym.orig_data is None or len(target_sym.orig_data) < min_data_points:
            len_ = None
            if target_sym.orig_data is not None:
                len_ = len(target_sym.orig_data)
            logger.info(f"target_sym[{target_sym.ticker} {target_sym.period}] [{len_}] has less than {min_data_points} data points.")
            # Change the last_analysis date to 5 days ago to force an update on the next analysis run.
            five_days_ago = datetime.now() - timedelta(days=5)
            target_sym.last_analysis = five_days_ago
            continue

        for ssym in sym_dpps.keys():

            if tsym != ssym:
                source_sym = sym_dpps[ssym]

                days_back = req_data_points
                if source_sym.period == PricePredict.PeriodWeekly:
                    days_back = req_data_points * 7

                if source_sym.orig_data is None or len(source_sym.orig_data) < min_data_points:
                    # Load up the data for the target symbol if it does not have enough data points
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    source_sym.fetch_data_yahoo(source_sym.ticker, start_date, end_date, source_sym.period)

                if source_sym.orig_data is None or len(source_sym.orig_data) < min_data_points:
                    len_ = None
                    if source_sym.orig_data is not None:
                        len_ = len(source_sym.orig_data)
                    logger.info(
                        f"1: Symbol source:[{ssym} {source_sym.period}] [{len_}] has less than {min_data_points} data points. Wont calculate correlations.")
                    continue

                corr = target_sym.periodic_correlation(source_sym)
                sym_corr[(tsym, ssym)] = corr

    corrs = []
    i = 0
    item_cnt = len(sym_corr)
    for ts in sym_corr.keys():
        i += 1
        # Update the progress bar
        prog_bar.progress(int(i / item_cnt * 100), f"{prd} Correlations (2): {tsym} ({i}/{item_cnt})")
        if sym_corr[ts] is None:
            logger.debug(f"Symbol Correlation (sym_corr) for [{ts}] is None")
            continue

        corrs.append((ts, (round(sym_corr[ts]['avg_corr'], 5),
                           round(sym_corr[ts]['pct_uncorr'], 5),
                           round(sym_corr[ts]['coint_test']['coint_measure'], 5),
                           round(sym_corr[ts]['coint_test']['p_val'], 5),)))

    # corrs[x] is the same as tup below
    # tup[x]
    indx_sym = 0
    indx_corr_vals = 1
    # tup[indx_sym][x]
    indx_trg_sym = 0
    indx_src_sym = 1
    # tup[indx_corr_vals][x]
    indx_avg_corr = 0
    indx_pct_uncorr = 1
    indx_coint_measure = 2
    indx_coint_p_val = 3

    # Sort the correlations by the pct_uncorr value
    # tup[1][0]: A symbols average correlation
    # tup[1][1]: A symbols percentage of uncorrelated data
    # tup[1][2]: A symbols cointegration p-value
    srt_coint = sorted(corrs, key=lambda tup: tup[indx_corr_vals][indx_coint_p_val], reverse=False)
    srt_corrs = sorted(corrs, key=lambda tup: tup[indx_corr_vals][indx_avg_corr], reverse=True)
    srt_xcorrs = sorted(corrs, key=lambda tup: tup[indx_corr_vals][indx_avg_corr], reverse=False)

    # For each symbol, get the top 10 correlations
    item_cnt = len(sym_dpps)
    i = 0
    for tsym in sym_dpps.keys():
        i += 1
        # Update the progress bar
        prog_bar.progress(int(i / item_cnt * 100), f"{prd} Correlations (3): {tsym} ({i}/{item_cnt})")
        # if just_one is not None and tsym != just_one:
        #     # Skip all symbols except the one specified
        #     continue
        target_sym = sym_dpps[tsym]
        sym_dpps[tsym].ticker = tsym  # Make sure the ticker is set correctly

        if target_sym.orig_data is None or len(target_sym.orig_data) < min_data_points:
            len_ = None
            if target_sym.orig_data is not None:
                len_ = len(target_sym.orig_data)
            logger.info(
                f"2: Symbol target:[{target_sym.ticker} {target_sym.period}] [{len_}] has less than {min_data_points} data points. Wont calculate correlations.")
            continue

        # Gather the top 10 cointegrated symbols for the target symbol
        top10coint = []

        j = 0
        for ssym in srt_coint:
            if j > 10:
                break
            if tsym == ssym[indx_sym][indx_trg_sym]:
                # (<other_sym>, <coint_measure>
                top10coint.append((ssym[indx_sym][indx_src_sym], ssym[indx_corr_vals][indx_coint_p_val]))
                j += 1
        target_sym.top10coint = top10coint

        # Gather the top 10 correlated symbols for the target symbol
        top10corr = []
        j = 0
        for ssym in srt_corrs:
            # ssym[][x]
            indx_avg_corr = 1
            if j > 10:
                break
            if tsym == ssym[indx_sym][indx_trg_sym]:
                # (<other_sym>, <coint_measure>
                top10corr.append((ssym[indx_sym][indx_src_sym], ssym[indx_corr_vals][indx_avg_corr]))
                j += 1
        target_sym.top10corr = top10corr

        # Gather the top 10 uncorrelated symbols for the target symbol
        top10xcorr = []
        j = 0
        for ssym in srt_xcorrs:
            # ssym[][x]
            indx_avg_corr = 1
            if j > 10:
                break
            if tsym == ssym[indx_sym][indx_trg_sym]:
                # (<other_sym>, <coint_measure>
                top10xcorr.append((ssym[indx_sym][indx_src_sym], ssym[indx_corr_vals][indx_avg_corr]))
                j += 1
        target_sym.top10xcorr = top10xcorr

    return


def model_cleanup(period, symbols):
    # Given the list of symbols, in sum_dpps_d...
    # Get the list of models that contain _D_ in the name...
    del_model_cnt = 0
    for sym in symbols['Symbol']:
        sym = sym.replace("=", "~")
        # Find all the files that match the symbol.
        syms_files = []
        for root, dirs, files in os.walk("./models"):
            for file in files:
                if file.startswith(sym) and period in file and file.endswith(".keras"):
                    syms_files.append(file)
        # Sort the file by file name in reverse order...
        syms_files = sorted(syms_files, reverse=True)
        # Remove all but the first file...
        for file in syms_files[1:]:
            os.remove(f"./models/{file}")
            del_model_cnt += 1

    return del_model_cnt


def ppo_cleanup(period, symbols):
    """ Delete all but the latest model files for the given period and symbols """
    syms_files = []
    for root, dirs, files in os.walk("./ppo"):
        for file in files:
            if file.endswith(".dill"):
                syms_files.append(file)
    syms_files = sorted(syms_files, reverse=True)
    del_model_cnt = 0
    for sym in symbols['Symbol']:
        sym = sym.replace("=", "~")
        sfiles = []
        for sf in syms_files:
            if sf.startswith(sym) and period in sf:
                if '_' + period + '_' in sf:
                    if sf not in sfiles:
                        sfiles.append(sf)
        # Find all the files that match the symbol.
        # Sort the file by file name in reverse order...
        sfiles = sorted(sfiles, reverse=True)
        # Remove all but the first file...
        for file in sfiles[1:]:
            fp = f"./ppo/{file}"
            if os.path.isfile(fp):
                os.remove(fp)
                del_model_cnt += 1

    return del_model_cnt


def charts_cleanup(st, period):
    # Remove the old charts...
    all_df_symbols = st.session_state[ss_AllDfSym]
    sym_list = all_df_symbols['Symbol'].values
    files_deleted = 0
    syms = []
    for root, dirs, files in os.walk("charts"):
        for file in files:
            if file.endswith(".png") and f"_{period}_" in file:
                fn_comps = file.split("_")
                fsym = fn_comps[0]
                fper = "_" + fn_comps[1] + "_"
                if fsym not in sym_list:
                    # Check if the charts file exists
                    if os.path.isfile(f"charts/{file}"):
                        # Remove the file if the symbol is not in the general symbol dataframe
                        os.remove(f"charts/{file}")
                        files_deleted += 1
                    continue
                if period in fper and fsym not in syms:
                    # Add files with the symbol in to the syms list
                    syms.append(fsym)

        files_in_charts = []
        for root, dirs, files in os.walk("charts"):
            for file in files:
                files_in_charts.append(file)

        for sym in syms:
            # Find all the files that match the symbol.
            syms_files = []
            for file in files_in_charts:
                if file.startswith(sym) and f'_{period}_' in file:
                    syms_files.append(file)
            # Sort the file by file name in reverse order...
            syms_files = sorted(syms_files, reverse=True)
            # Remove all but the first file...
            if len(syms_files) > 1:
                for file in syms_files[1:]:
                    # Check if the charts file exists
                    if os.path.isfile(f"charts/{file}"):
                        os.remove(f"charts/{file}")
                        files_deleted += 1

    return files_deleted


def do_bayes_opt(in_ticker, pp_obj=None, opt_csv=None,
                  only_fetch_opt_data=False, do_optimize=False,
                  cache_train=False, cache_predict=False, train_and_predict=False):
    if pp_obj is None:
        # Create an instance of the price prediction object
        pp = PricePredict(model_dir=model_dir,
                          chart_dir=chart_dir,
                          preds_dir=preds_dir)
    else:
        pp = pp_obj

    # Load data from Yahoo Finance
    ticker = in_ticker
    # Training Data (Training uses 20% of the latest data for validation)
    end_dt = datetime.now()
    start_dt = (end_dt - timedelta(days=365 * 4))
    end_date = end_dt.strftime("%Y-%m-%d")
    start_date = start_dt.strftime("%Y-%m-%d")
    # Prediction Data
    pred_end_dt = datetime.now()
    pred_start_dt = (pred_end_dt - timedelta(days=30 * 3))
    pred_end_date = pred_end_dt.strftime("%Y-%m-%d")
    pred_start_date = pred_start_dt.strftime("%Y-%m-%d")

    if only_fetch_opt_data:
        data, pp.features = pp.fetch_data_yahoo(ticker, start_date, end_date)

        # Augment the data with additional indicators/features
        if data is None:
            print(f"'Close' column not found in {ticker}'s data. Skipping...")
            return None

        return pp

    if do_optimize:
        aug_data, features, targets, dates_data = pp.augment_data(pp.orig_data, 0)

        # Scale the data so the model can use it more effectively
        scaled_data, scaler = pp.scale_data(aug_data)

        # Prepare the scaled data for model inputs
        X, y = pp.prep_model_inputs(scaled_data, pp.features)

        # Train the model
        model, y_pred, mse = pp.train_model(X, y)

        # Perform Bayesian optimization
        pp.bayesian_optimization(X, y, opt_csv=opt_csv)

    if cache_train:
        pp.cache_training_data(ticker, start_date, end_date, PricePredict.PeriodDaily)
    if cache_predict:
        pp.cache_prediction_data(ticker, pred_start_date, pred_end_date, PricePredict.PeriodDaily)
    if train_and_predict:
        # Training, will load last saved model which is the optimized model.
        pp.cached_train_predict_report(force_training=False, save_plot=False, show_plot=True)

    return pp


def optimize_hparams(st, prog_bar):

    # Read ./gui_data/gui_all_symbols.csv into a dataframe
    import pandas as pd
    df_tickers = pd.read_csv(guiAllSymbolsCsv)

    already_optimized = {}
    with open(opt_hyperparams, 'r') as f:
        for line in f:
            if line is not None and line.strip() != '':
                opt_params = json.loads(line)
                sym = opt_params['symbol']
                print(f"---> Sym: {sym}")
                already_optimized[sym] = opt_params

    ticker_pp = {}
    futures = []
    opt_cnt = 1

    with cf.ThreadPoolExecutor(10) as ex:
        for ticker in df_tickers['Symbol']:
            if ticker in already_optimized and not st.session_state[ss_forceOptimization]:
                continue
            # Sync: Pull in Training and Prediction Data for each Ticker
            print(f"Pulling Optimization data for {ticker}...")
            pp = do_bayes_opt(ticker, only_fetch_opt_data=True)
            ticker_pp[ticker] = pp
        opt_cnt = 0
        for ticker in df_tickers['Symbol']:
            opt_cnt += 1
            # Update the progress bar
            prog_bar.progress(int(opt_cnt / len(df_tickers) * 100),
                              f"Loading Model Data: {ticker} ({opt_cnt}/{len(df_tickers)})")

            if ticker in already_optimized and not st.session_state[ss_forceOptimization]:
                continue
            # Async: Optimize the Model's Hyperparameters for each Ticker
            print(f"Optimizing model for {ticker}...")
            pp = ticker_pp[ticker]
            kawrgs={'pp_obj': pp, 'do_optimize': True}
            future = ex.submit(do_bayes_opt, ticker, **kawrgs)
            futures.append(future)

        if len(futures) == 0:
            st.info("No optimization tasks were created.")

        print("Waiting for tasks to complete...")
        write_mode = 'a'
        if st.session_state[ss_forceOptimization]:
            write_mode = 'w'
        with open(opt_hyperparams, write_mode) as f:
            opt_cnt = 0
            for future in cf.as_completed(futures):
                try:
                    pp = future.result()
                except Exception as e:
                    print(f"Optimization for {ticker} generated an exception: {e}")
                else:
                    # Write out the optimized hyperparameters to a JSON file
                    opt_hypers_s = json.dumps(pp.bayes_opt_hypers)
                    f.write(f'{{ "symbol": "{pp.ticker}", "hparams": {opt_hypers_s} }}\n')
                    print(f'Completed Hyperparameters Optimization: {pp.ticker}')
                opt_cnt += 1
                # Update the progress bar
                prog_bar.progress(int(opt_cnt / len(futures) * 100),
                                  f"Completed Model Optimization: {pp.ticker} ({opt_cnt}/{len(futures)})")

    if opt_cnt > 0:
        st.info(f"All [{opt_cnt}] optimization tasks completed.")

    print("All optimization tasks completed.")


def review_pp_objects(st, period):
    if period == PricePredict.PeriodDaily:
        sym_dpps = st.session_state[ss_SymDpps_d]
    elif period == PricePredict.PeriodWeekly:
        sym_dpps = st.session_state[ss_SymDpps_w]
    else:
        logger.error(f"Error: Invalid period: {period}")
        return

    for sym in sym_dpps:
        ppo = sym_dpps[sym]
        # Get first file in './ppo/save' directory that starts with the symbol in sym.
        files = os.listdir('./ppo/save')
        ppo_save_file = None
        for file in files:
            if file.startswith(sym):
                ppo_save_file = file
                break
        if ppo_save_file is not None:
            # Unpickle the PricePredict object
            ppo_save = None
            with open(f'./ppo/save/{ppo_save_file}', 'rb') as f:
                try:
                    ppo_save = dill.load(f)
                except Exception as e:
                    logger.error(f"Error loading PricePredict object: {e}")
            if ppo_save is not None:
                # Compare cached_train_data and cached_predict_data properties
                # between ppo and ppo_saved objects after converting them into JSON strings.
                ppo_json = json.dumps(ppo.cached_train_data)
                ppo_save_json = json.dumps(ppo_save.cached_train_data)
                if ppo_json != ppo_save_json:
                    logger.error(f"Error: {sym} {period} cached_train_data objects do not match.")
                    logger.error(f"ppo: {ppo_json}")
                    logger.error(f"ppo_save: {ppo_save_json}")
                ppo_json = json.dumps(ppo.cached_pred_data)
                ppo_save_json = json.dumps(ppo_save.cached_pred_data)
                if ppo_json != ppo_save_json:
                    logger.error(f"Error: {sym} {period} cached_pred_data objects do not match.")
                    logger.error(f"ppo: {ppo_json}")
                    logger.error(f"ppo_save: {ppo_save_json}")
        else:
            logger.error(f"Error: {sym} {period} PricePredict object not found.")


if __name__ == "__main__":
    my_msg = "I'm Still Here!"
    main(my_msg)
    # profiler = LineProfiler()
    # profiler.add_function(main)
    # profiler.add_function(analyze_symbols)
    # profiler.add_function(task_pull_data)
    # profiler.add_function(task_train_predict_report)
    # profiler.add_function(update_viz_data)
    # profiler.add_function(sym_correlations)
    # profiler.add_function(st_dataframe_widget)
    # profiler.runcall(main, my_msg)
    # profiler.print_stats()
