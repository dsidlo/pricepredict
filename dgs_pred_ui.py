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

import json
import time
import streamlit as st
import pandas as pd
import os
import logging
import dill
import shutil
import yfinance as yf
import futureproof as fp
import hashlib
import re

from datetime import datetime, timedelta
from pricepredict import PricePredict
from io import StringIO

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
ss_bCancelTglFaves = 'ss_bCancelTglFaves'
ss_bToggleGroups = 'ss_bToggleGroups'
ss_bCancelRmSyms = 'cancel_rm_syms'
ss_bRemoveChosenSyms = 'remove_chosen_syms'
# Flags...
ss_fDf_symbols_updated = 'st_df_symbols_updated'
ss_fRmChosenSyms = 'Remove Chosen Symbols Flag'
ss_fTglFavs = 'Toggle Favorites'
ss_forceTraining = "cbForceTraining"
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
sym_dpps_d = None
sym_dpps_w = None
# Directory for gui files
gui_data = 'gui_data'
# Save/Restore file for the all_df_symbols DataFrame
guiAllSymbolsCsv = f'{gui_data}/gui_all_symbols.csv'
# Pickle files for the PP objects
dill_sym_dpps_d = f'{gui_data}/sym_dpps_d.dil'
dill_sym_dpps_w = f'{gui_data}/sym_dpps_w.dil'
dillbk_sym_dpps_d = f'{gui_data}/sym_dpps_d.dil.bk'
dillbk_sym_dpps_w = f'{gui_data}/sym_dpps_w.dil.bk'

import_cols_simple = 2
import_cols_full = 10

# ======================================================================
# Main Window ==========================================================
# The main window displays the charts and data for the selected symbol.
# ======================================================================
def main(message):

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
            # Load the symbols from the file
            st.session_state[ss_AllDfSym] = pd.read_csv(guiAllSymbolsCsv, header='infer')
            # Load pp daily objects
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
        st.sidebar.title("*Price Prediction*")
        st.sidebar.markdown("## Symbols")

        # Add Groups ==========================================
        if ss_GroupsList not in st.session_state:
            st.session_state[ss_GroupsList] = []
        # Add Expander for adding and removing groupings
        exp_grps = st.expander("Add/Remove Groups", expanded=False)
        col_add_grp1, col_add_grp2 = exp_grps.columns(2)
        col_add_grp1.text("Add New Group")
        ti_addNewGrp = col_add_grp2.text_input("Add New Group", label_visibility='collapsed',
                                               key=tiNewGrp, value='', placeholder="Enter a new Group")
        if hasattr(st.session_state, bRremoveGroup):
            if st.session_state.tiNewGrp != '' and st.session_state.tiNewGrp not in st.session_state[ss_GroupsList]:
                # Remove the Group from the Groups List
                st.session_state[ss_GroupsList].append(st.session_state.tiNewGrp)
                st.session_state[ss_GroupsList] = list(set(st.session_state[ss_GroupsList]))
                st.session_state[ss_GroupsList].sort()
                ti_addNewGrp = st.empty()

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

        # -- Create a Column Container for symbol operations and the symbol list
        col_tgl_grp1, col_tgl_grp2 = exp_grps.columns(2)
        # -- Add a button for initiating toggle favorites
        # Init the "Toggle Groups" state flag.
        if ss_fTglFavs not in st.session_state:
            st.session_state[ss_fTglFavs] = False
        toggle_chosen_faves = st.session_state[ss_fTglFavs]
        col_tgl_grp1.button("Toggle Groups", key=ss_bToggleGroups)
        col_tgl_grp2.selectbox("Set Group", label_visibility='collapsed',
                               options=st.session_state[ss_GroupsList],
                               key=sbToggleGrp)

        # Init the "Remove Chosen Symbols" state flag.
        if ss_fRmChosenSyms not in st.session_state:
            st.session_state[ss_fRmChosenSyms] = False
        rm_chosen_syms = st.session_state[ss_fRmChosenSyms]

        # -- Add Expander for adding a new symbol
        exp_sym = st.expander("Add/Remove Symbols", expanded=False)
        # -- Add a text input for adding a new symbol
        if 'new_sym_value' not in locals():
            new_sym_value = ''
        ti_addNewSyms = exp_sym.text_input("Add New Symbols", key="new_sym",
                                           value='', placeholder="Enter a new symbol")
        imported_data = b''
        # -- Add a file uploader button for importing symbols
        bt_import_syms = "import_syms"
        exp_sym.file_uploader("Import Symbols", key="import_syms",
                                type='csv')
        # -- Add a button for initiating removing the imported symbols
        exp_sym.button("Remove Imported Symbols", key="remove_imp_sym")
        if (hasattr(st.session_state, ss_fDf_symbols_updated) and st.session_state[ss_fDf_symbols_updated]
            # Indicate successful removal of imported symbols
            and st.session_state['imp_sums_removed'] > 0):
            exp_sym.success(f'**[{st.session_state["imp_sums_removed"]}] Imported Symbols Removed.**', icon='👍')
            st.session_state[ss_fDf_symbols_updated] = False

        # -- Add a button for adding the new symbol
        exp_sym.button("Choose Symbols to Remove", key="bRemove_sym")

        # -- Add a button for process the current symbols
        col1, col2 = exp_sym.columns(2)
        col1.button("Analyze Current Symbols", key="process_syms")
        col2.checkbox("Force Training", key=ss_forceTraining)
        if st.session_state.process_syms:
            # This code results in updates to the progress bar but
            # the main thread blocks until the processing is completes.
            # "await" causes the block. Can we run this in a fragment?
            prog_bar = exp_sym.progress(0, "Analyzing Symbols")
            # await analyze_symbols(prog_bar, st.session_state[ss_DfSym])
            analyze_symbols(st, prog_bar, st.session_state[ss_DfSym])

        elif 'prog_bar' in locals():
            prog_bar.empty()

        if st.session_state.import_syms is not None and not st.session_state.bRemove_sym:
            import_symbols(st, exp_sym)

        st.sidebar.add_rows = exp_sym

        # -- Handle add new symbols text input
        if ti_addNewSyms != '':
            # Split on new_sym on spaces, colon, semicolon and commas
            new_syms = ti_addNewSyms
            new_syms = re.split(r'[\s,;:]+', new_syms)
            if new_syms[0] != '':
                add_new_symbols(st, exp_sym, new_syms)
            # Save all_df_symbols
            all_df_symbols = st.session_state[ss_AllDfSym]
            df_symbols = st.session_state[ss_DfSym]
            merge_and_save(all_df_symbols, df_symbols)
            ti_addNewSyms.replace(ti_addNewSyms, '')

        # -- Add Expander for filtering the symbols
        exp_sort_filter = st.expander("Filter", expanded=False)
        # -- Add a dropdown for filtering the symbols
        exp_sort_filter.selectbox(
            # Add a dropdown for filtering the symbols
            "***- filter -***",
            ("All", "Up Trending", "Down Trending", "Flat", "Imported"),
            on_change=filter_symbols,
            # Make this elements value available via the st.session_state object.
            key="filter_sym"
        )

        # -- Create a Column Container for symbol operations and the symbol list
        sym_col = st.columns(1)

        # Single-Row / Mylti-Row Toggle :  =================
        if ((st.session_state.bRemove_sym or rm_chosen_syms)               # Remove Chosen Symbols
            or (st.session_state.sbToggleGrp or toggle_chosen_faves)):   # Mark/Clear Favorites
            logger.info("2>>> Multi-Row on")
            if st.session_state.bRemove_sym or rm_chosen_syms:   # Remove Chosen Symbols
                st.session_state[ss_fRmChosenSyms] = True
            if st.session_state.sbToggleGrp or toggle_chosen_faves:   # Mark/Clear Favorites
                st.session_state[ss_fTglFavs] = True
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
        elif st.session_state.sbToggleGrp != '' or toggle_chosen_faves:
            # Remove the imported symbols from the DataFrame
            logger.info("*** Toggle Favorites ***")
            st.session_state[ss_fTglFavs] = True
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

        # Remove Imported Symbols =====================================
        if hasattr(st.session_state, 'remove_imp_sym') and st.session_state.remove_imp_sym:
            dlg_rm_imp_syms()

        if hasattr(st.session_state, 'yes_remove_imp_sym') and st.session_state.yes_remove_imp_sym:
            if hasattr(st.session_state, ss_fDf_symbols_updated) and st.session_state[ss_fDf_symbols_updated]:
                st.session_state[ss_fDf_symbols_updated] = False
                st.rerun()
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
                # Remove the selected symbol from the sym_dpps_w and sym_dpps_d dictionaries
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
            st.rerun()
        if hasattr(st.session_state, ss_bCancelRmSyms) and st.session_state.cancel_rm_syms:
            st.session_state[ss_fRmChosenSyms] = False  # Turns off the Action Buttons
            logger.info('*** Canceled: Remove Chosen Symbols ***')
            st.rerun()
        # =====================================================

        # Toggle Favorites ====================================
        if hasattr(st.session_state, ss_bToggleGroups) and st.session_state.ss_bToggleGroups:
            st.session_state[ss_fTglFavs] = False   # Turns off the Action Buttons
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
                st.session_state[ss_fTglFavs] = False   # Turns off the Action Buttons
                logger.info('*** Canceled: Remove Chosen Symbols ***')
                st.rerun()  # Refresh the page
        # =====================================================

    display_symbol_charts()

def merge_and_save(all_df_symbols, df_symbols):
    # Merge df_symbols with all_df_symbols
    for row in df_symbols.itertuples():
        # Get the index to the existing symbol
        idx = all_df_symbols.index[all_df_symbols.Symbol == row.Symbol]
        if idx is not None:
            all_df_symbols.loc[idx] = [row.Symbol, row.LongName,
                                       row.Groups, row.Trend,
                                       row.WklyPrdStg, row.DlyPrdStg,
                                       row.wTop10Corr, row.wTop10xCorr,
                                       row.dTop10Corr, row.dTop10xCorr]
        elif len(idx) == 0:
            # Add the symbol to the DataFrame
            all_df_symbols.loc[len(all_df_symbols)] = [row.Symbol, row.LongName,
                                                       row.Groups, row.Trend,
                                                       row.WklyPrdStg, row.DlyPrdStg,
                                                       row.wTop10Corr, row.wTop10xCorr,
                                                       row.dTop10Corr, row.dTop10xCorr]
    all_df_symbols.sort_values("Symbol", inplace=True)
    all_df_symbols.reindex()
    st.session_state[ss_AllDfSym] = all_df_symbols
    # Save out the updated DataFrames and PricePredict objects
    if len(all_df_symbols.columns) == import_cols_full:
        # Sort all_df_symbols by Symbol
        all_df_symbols.to_csv(guiAllSymbolsCsv, index=False)
    else:
        logger.error(
            f"Error all_df_symbols has too many columns [{len(all_df_symbols)}]: {all_df_symbols.columns}")


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
            or (hasattr(st.session_state, ss_fTglFavs)
                and st.session_state[ss_fTglFavs])
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
    # if (st.session_state.bRemove_sym and hasattr(st.session_state, ss_fRmChosenSyms) is False)\
    #         or (hasattr(st.session_state, ss_fRmChosenSyms) and st.session_state[ss_fRmChosenSyms] is False):
    print(f' *** df_sel_mode: {df_sel_mode}, on_select: {on_select}')
    print(f' *** ss_fRmChosenSyms [{hasattr(st.session_state, ss_fRmChosenSyms)}]')
    # Fill null values in the Groups column with an empty string
    st.session_state[ss_DfSym]['Groups'] = st.session_state[ss_DfSym]['Groups'].fillna('')
    st.session_state[ss_DfSym] = st.session_state[ss_DfSym].ffill()
    st.session_state[ss_DfSym] = st.session_state[ss_DfSym].bfill()
    df_symbols = st.session_state[ss_DfSym].copy()
    df_styled = df_symbols.style.format({'WklyPrdStg': '{:,.4f}',
                                         'DlyPrdStg': '{:,.4f}'})
    # df_styled = st.session_state[ss_DfSym]
    df_sym = st.dataframe(df_styled, key="dfSymbols", height=600,
                          column_order=["Symbol", "Groups", "Trend",
                                        "WklyPrdStg", "DlyPrdStg"],
                          column_config={"Symbol": {"max_width": 5},
                                         "Groups": {"max_width": 4},
                                         "Trend": {"max_width": 5},
                                         "WklyPrdStg": {"max_width": 5},
                                         "DlyPrdStg": {"max_width": 5}},
                          selection_mode=df_sel_mode, on_select=on_select, hide_index=True)
    sym_col.append(df_sym.selection)
    # Display Chosen Symbol's Chart & Data ========================
    if hasattr(df_sym.selection, 'rows') and len(df_sym.selection.rows) > 0:
        # We have a selected symbol so display the symbol's chart
        df_symbols = st.session_state[ss_DfSym]
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
        logger.debug(f"3> Saving all_df_symbols to {guiAllSymbolsCsv}")
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
    # Remove the selected symbol from the sym_dpps_w and sym_dpps_d dictionaries
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
        df_imported_syms['Trend'] = ' '
        df_imported_syms['WklyPrdStg'] = 0.0
        df_imported_syms['DlyPrdStg'] = 0.0
        df_imported_syms['wTop10Corr'] = ' '
        df_imported_syms['wTop10xCorr'] = ' '
        df_imported_syms['dTop10Corr'] = ' '
        df_imported_syms['dTop10xCorr'] = ' '

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
    for sym in syms:
        # Verify with yahoo finance that the symbol is valid, and get the long name
        logger.info(f"1. New Symbols to be added: {st.session_state.new_sym}")
        if sym == '':
            continue
        if sym not in st.session_state[ss_DfSym].Symbol.values:
            # Verify with yahoo finance that the symbol is valid, and get the long name
            new_ticker, long_name = getTickerLongName(sym)
            if new_ticker is not None:
                new_row = pd.DataFrame({'Symbol': sym,
                                        'LongName': long_name,
                                        'Groups': 'Imported', 'Trend': '',
                                        'DlyPrdStg': [0.0], 'WklyPrdStg': [0.0],
                                        'wTop10Corr': [''], 'wTop10xCorr': [''],
                                        'dTop10Corr': [''], 'dTop10xCorr': ['']})
                df_symbols = pd.concat([df_symbols, new_row],
                                       ignore_index=True)
                all_df_symbols = pd.concat([all_df_symbols, new_row], ignore_index=True)
                added_symbols.append(sym)
        else:
            already_exists.append(sym)

    if len(added_symbols) == 0:
        st.info(f"Symbols Already Exists: {already_exists}")
        return

    df_symbols.reindex()
    st.session_state[ss_DfSym] = df_symbols
    all_df_symbols.reindex()
    st.session_state[ss_AllDfSym] = all_df_symbols

    for sym in added_symbols:
        # Create a datetime that is 5 days ago
        five_days_ago = datetime.now() - timedelta(days=5)
        pp_d = PricePredict(ticker=sym, period=PricePredict.PeriodDaily)
        pp_d.last_analysis = five_days_ago
        st.session_state[ss_SymDpps_d][sym] = pp_d
        pp_w = PricePredict(ticker=sym,
                            period=PricePredict.PeriodWeekly)
        pp_w.last_analysis = five_days_ago
        st.session_state[ss_SymDpps_w][sym] = pp_w

    txt = f"New Symbols Added: {added_symbols}"
    if len(already_exists) > 0:
        txt += f"\nSymbols Already Exists: {already_exists}"
    st.info(f"New Symbols Added: {added_symbols}")

    # Run an analysis on all symbols
    prog_bar = exp_sym.progress(0, "Analyzing Symbols")
    # await analyze_symbols(prog_bar, st.session_state[ss_DfSym])
    analyze_symbols(st, prog_bar, st.session_state[ss_DfSym],
                    imported_syms=added_symbols)


def display_symbol_charts():
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

    img_cols = st.columns([1], gap='small', vertical_alignment='center')
    with img_cols[0]:
        # Ticker Chosen: Show tickers charts and data
        if 'img_sym' in locals() and img_sym is not None:
            # if ss_DfSym in st.session_state and 'Symbol' in st.session_state[ss_DfSym]:
            ss_df = st.session_state[ss_DfSym]
            sym_longName = ss_df.loc[ss_df['Symbol'] == img_sym, 'LongName'].iloc[0]
            # Within the main window...
            st.markdown(f"## - {img_sym}: {sym_longName} -")
            st.markdown(f"### Weekly Chart")
            w_img_file = get_sym_image_file(img_sym, PricePredict.PeriodWeekly, "charts")
            if w_img_file is not None:
                if today not in w_img_file:
                    st.markdown("#### *** Chart May Not Be Current ***")
                # Display the daily chart image
                try:
                    st.image(w_img_file, use_column_width='auto')
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
            st.markdown(f"### Daily Chart")
            d_img_file = get_sym_image_file(img_sym, PricePredict.PeriodDaily, "charts")
            if d_img_file is not None:
                if today not in d_img_file:
                    st.markdown("#### *** Chart May Not Be Current ***")
                # Display the daily chart image
                try:
                    st.image(d_img_file, use_column_width='auto')
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

            st.markdown(f"=====   =====   =====   =====   ====   ====   ====    ====   ====   =====   =====   =====   =====   =====   =====   =====   =====")

            if img_sym not in st.session_state[ss_SymDpps_d]:
                logger.error(f"Symbol [{img_sym}] not found in PricePredict objects")
            else:
                pp = st.session_state[ss_SymDpps_d][img_sym]
                expdr_corr = st.expander("**Correlations**", expanded=False)
                col1, col2 = expdr_corr.columns(2)

                col1.markdown('**Top 10 Correlated**')
                df_to10corr = pd.DataFrame(data=pp.top10corr, columns=['Symbol', 'Correlation'])
                styl_to10corr = df_to10corr.style.set_properties(**{'boarder': '2px solid #ccc'})
                col1.table(styl_to10corr)
                # col1.table(pp.top10corr)

                col2.markdown('**Top 10 X-Correlated**')
                df_to10xcorr = pd.DataFrame(data=pp.top10xcorr, columns=['Symbol', 'Correlation'])
                styl_to10xcorr = df_to10xcorr.style.set_properties(**{'boarder': '2px solid #ccc'})
                col2.table(styl_to10xcorr)
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
    # As needed, save out the updated DataFrames and PricePredict objects
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
            logger.debug(f"1> Saving all_df_symbols to {guiAllSymbolsCsv}")
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
def analyze_symbols(st, prog_bar, df_symbols, just_one=None, imported_syms=None):
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
            if row.Symbol == 'Index':
                # Skip the header row
                logger.info(f"Skipping header row [{row.Symbol}]")
                continue

            if just_one is not None and row.Symbol != just_one:
                # Skip all symbols except the one specified
                continue
            if just_one is not None:
                logger.info(f"just_one: processing symbol [{row.Symbol}]")

            if imported_syms is not None and row.Symbol not in imported_syms:
                # Skip all symbols not in the imported symbols list
                continue
            else:
                logger.info(f"imported_syms/added_sym: processing symbol [{row.Symbol}]")
            # Check if the model has a weekly PricePredict object
            if row.Symbol not in st.session_state[ss_SymDpps_w]:
                # Create a weekly PricePredict object for the symbol
                ppw = PricePredict(ticker=row.Symbol, period=PricePredict.PeriodWeekly)
                st.session_state[ss_SymDpps_w][row.Symbol] = ppw
                st.session_state[ss_SymDpps_w][row.Groups] = 'Imported'
            else:
                ppw = st.session_state[ss_SymDpps_w][row.Symbol]

            if st.session_state[ss_forceTraining]:
                ppw.orig_data = None
            if ppw.orig_data is None:
                # Get the datetime of 5 days ago
                five_days_ago = datetime.now() - timedelta(days=7)
                # This will force an update of the ppw object
                ppw.last_analysis = five_days_ago
                ppw.force_training = st.session_state[ss_forceTraining]

            logger.info(f"Weekly - Pull data for model: {row.Symbol}")
            try:
                task_pull_data(row.Symbol, ppw)
                logger.info(f"Weekly - Train and Predict: {row.Symbol}")
                # future = executor.submit(task_train_predict_report, row.Symbol, ppw)
                # futures.append(future)
                tm.submit(task_train_predict_report, row.Symbol, ppw)
                total_syms += 1
            except Exception as e:
                logger.error(f"Error processing symbol: {row.Symbol}\n{e}")

            # Check if the model has a daily PricePredict object
            if row.Symbol not in st.session_state[ss_SymDpps_d]:
                # Create a daily PricePredict object for the symbol
                ppd = PricePredict(ticker=row.Symbol, period=PricePredict.PeriodDaily)
                st.session_state[ss_SymDpps_d][row.Symbol] = ppd
            else:
                ppd = st.session_state[ss_SymDpps_d][row.Symbol]

            if st.session_state[ss_forceTraining]:
                ppd.orig_data = None
            if ppd.orig_data is None:
                # Get the datetime of 5 days ago
                five_days_ago = datetime.now() - timedelta(days=3)
                # This will force an update of the ppd object
                ppd.last_analysis = five_days_ago
                ppd.force_training = st.session_state[ss_forceTraining]

            logger.info(f"Daily - Pull data for model: {row.Symbol}")
            try:
                task_pull_data(row.Symbol, ppd)
                logger.info(f"Daily - Train and Predict: {row.Symbol}")
                # future = executor.submit(task_train_predict_report, row.Symbol, ppd)
                # futures.append(future)
                tm.submit(task_train_predict_report, row.Symbol, ppd)
                total_syms += 1
            except Exception as e:
                logger.error(f"Error pull-train-predict on symbol: {row.Symbol}\n{e}")

            i += 1
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Analyzing: {row.Symbol} ({i}/{total_syms})")

        logger.debug(f"=== Checking on completed futures ...")
        # for future in concurrent.futures.as_completed(futures):
        for future in tm.as_completed():
            logger.debug(f"=== Checking futures ...")

            i += 1
            # Update the progress bar
            prog_bar.progress(int(i / total_syms * 100), f"Analyzing: {row.Symbol} ({i}/{total_syms})")

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

    sym_correlations('Weekly', st, st.session_state[ss_SymDpps_w], prog_bar,
                     just_one=just_one)
    sym_correlations('Daily', st, st.session_state[ss_SymDpps_d], prog_bar,
                     just_one=just_one)

    # Push all_df_symbols data into the df_symbols DataFrame for display
    for row in all_df_symbols.itertuples():
        idx = all_df_symbols.index[df_symbols.Symbol == row.Symbol].tolist()
        if len(idx) == 1:
            df_symbols.loc[idx[0], 'Trend'] = row.Trend
            df_symbols.loc[idx[0], 'WklyPrdStg'] = row.WklyPrdStg
            df_symbols.loc[idx[0], 'DlyPrdStg'] = row.DlyPrdStg
            df_symbols.loc[idx[0], 'wTop10Corr'] = row.wTop10Corr
            df_symbols.loc[idx[0], 'wTop10xCorr'] = row.wTop10xCorr
            df_symbols.loc[idx[0], 'dTop10Corr'] = row.dTop10Corr
            df_symbols.loc[idx[0], 'dTop10xCorr'] = row.dTop10xCorr

    st.session_state[ss_AllDfSym] = update_viz_data(st, all_df_symbols)

    # Save out the updated DataFrames and PricePredict objects
    store_pp_objects(st)
    logger.info("--- Analyzing Symbols: Completed ---")


def load_pp_objects(st):
    sym_dpps_d_ = {}
    sym_dpps_w_ = {}

    min_dil_size = 70000

    # Check if the PricePredict objects files exist
    if not os.path.exists(dill_sym_dpps_d) or not os.path.exists(dill_sym_dpps_w):
        logger.error("PricePredict object files do not exist")
        st.session_state[ss_SymDpps_d] = sym_dpps_d_
        st.session_state[ss_SymDpps_w] = sym_dpps_w_
        sync_dpps_objects(st)
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
        sym_dpps_d_['___'] = PricePredict(ticker='___', period=PricePredict.PeriodDaily)
    if sym_dpps_w_ is None:
        # Add a dummy PricePredict object so that other logic doesn't break
        sym_dpps_w_['___'] = PricePredict(ticker='___', period=PricePredict.PeriodWeekly)

    st.session_state[ss_SymDpps_d] = sym_dpps_d_
    st.session_state[ss_SymDpps_w] = sym_dpps_w_

    sync_dpps_objects(st)

    return sym_dpps_d_, sym_dpps_w_


def sync_dpps_objects(st):
    logger.info("Remove PricePredict objects that are not in the DataFrame")
    df_symbols = st.session_state[ss_AllDfSym]
    if ss_SymDpps_d not in st.session_state.keys() or ss_SymDpps_w not in st.session_state.keys():
        return
    if st.session_state[ss_SymDpps_d] is None or st.session_state[ss_SymDpps_w] is None:
        return
    sym_dpp_d = st.session_state[ss_SymDpps_d].copy()
    sym_dpp_w = st.session_state[ss_SymDpps_w].copy()
    for sym in sym_dpp_d:
        if sym not in df_symbols.Symbol.values:
            del st.session_state[ss_SymDpps_d][sym]
    for sym in sym_dpp_w:
        if sym not in df_symbols.Symbol.values:
            del st.session_state[ss_SymDpps_w][sym]

    # Make sure that we have PricePredict objects for all the symbols in the DataFrame
    for sym in df_symbols.Symbol.values:
        if sym not in st.session_state[ss_SymDpps_d]:
            # Create missing PricePredict objects
            pp = PricePredict(ticker=sym, period=PricePredict.PeriodDaily, logger=logger)
            st.session_state[ss_SymDpps_d][sym] = pp
        if sym not in st.session_state[ss_SymDpps_w]:
            # Create missing PricePredict objects
            pp = PricePredict(ticker=sym, period=PricePredict.PeriodWeekly, logger=logger)
            st.session_state[ss_SymDpps_w][sym] = pp


def store_pp_objects(st):
    sync_dpps_objects(st)
    logger.info("Saving PricePredict objects")
    st.info("Saving PricePredict objects...")
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

    logger.info("Saving PricePredict objects")
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
        st.error(f"Error saving PricePredict objects: {e}")


def task_pull_data(symbol_, dpp):
    # Get datetime 24 hours ago
    ago24hrs = datetime.now() - timedelta(days=1)

    logger.info(f"Pulling data for {symbol_}...")
    if dpp.last_analysis is not None:
        if dpp.last_analysis > ago24hrs:
            logger.info(f"PricePredict object is already updodate: {symbol_}")
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
    start_date = (datetime.strptime(end_date, "%Y-%m-%d")
                  - timedelta(days=160)).strftime("%Y-%m-%d")
    # Simply pulls and caches the prediction data...
    dpp.cache_prediction_data(symbol_, start_date, end_date, dpp.period)
    logger.info(f"Completed pulling data for {symbol_}...")
    return symbol_, dpp


def task_train_predict_report(symbol_, dpp):
    # Get datetime 24 hours ago
    ago24hrs = datetime.now() - timedelta(days=1)

    logger.info(f"Training and predicting for {symbol_}...")
    if dpp.last_analysis is not None:
        if dpp.last_analysis > ago24hrs:
            logger.info(f"PricePredict object is already updodate: {symbol_}")
            return symbol_, dpp
    # Process the cached data as needed...
    # - Trains and Saves a new model if needed
    # - Performs the prediction on the cached prediction data
    # - Generates the required charts and database updates
    dpp.cached_train_predict_report()
    logger.info(f"Completed training and predicting for {symbol_}...")
    return symbol_, dpp


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def update_viz_data(st, all_df_symbols):

    min_data_points = 50

    if ss_SymDpps_d in st.session_state.keys():
        sym_dpps_d = st.session_state[ss_SymDpps_d]
        all_df_symbols = st.session_state[ss_AllDfSym]
        for sym in all_df_symbols.Symbol.values:
            if sym in sym_dpps_d:
                pp = sym_dpps_d[sym]
            else:
                pp = PricePredict(ticker=sym, period=PricePredict.PeriodDaily, logger=logger)
                sym_dpps_d[sym] = pp

            # if pp.orig_data is None or len(pp.orig_data) < min_data_points:
            #     continue
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
            trend = 'D:'
            if pp.pred_strength is not None:
                if abs(pp.pred_strength) < 0.5:
                    trend += 'f'
                elif pp.pred_strength < 0:
                    trend += 'd'
                else:
                    trend += 'u'
            else:
                trend += '_'

            all_df_symbols.loc[indx, 'Trend'] = trend
            prd_strg = pp.pred_strength
            if prd_strg is None:
                prd_strg = 0
            all_df_symbols.loc[indx, 'DlyPrdStg'] = prd_strg
            all_df_symbols.loc[indx, 'dTop10Corr'] = json.dumps(pp.top10corr)
            all_df_symbols.loc[indx, 'dTop10xCorr'] = json.dumps(pp.top10xcorr)

    if ss_SymDpps_w in st.session_state.keys():
        sym_dpps_w = st.session_state[ss_SymDpps_w]
        for sym in all_df_symbols.Symbol.values:
            if sym in sym_dpps_w:
                pp = sym_dpps_w[sym]
            else:
                pp = PricePredict(ticker=sym, period=PricePredict.PeriodWeekly, logger=logger)
                sym_dpps_w[sym] = pp

            # if not hasattr(pp, 'orig_data') or pp.orig_data is None or len(pp.orig_data) < min_data_points:
            #     continue

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
            trend = all_df_symbols[all_df_symbols.Symbol == sym]['Trend'].values[0] + ' - W:'
            if pp.pred_strength is not None:
                if abs(pp.pred_strength) < 0.5:
                    trend += 'f'
                elif pp.pred_strength < 0:
                    trend += 'd'
                else:
                    trend += 'u'
            else:
                trend += '_'

            all_df_symbols.loc[indx, 'Trend'] = trend
            prd_strg = pp.pred_strength
            if prd_strg is None:
                prd_strg = 0
            all_df_symbols.loc[indx, 'WklyPrdStg'] = prd_strg
            all_df_symbols.loc[indx, 'wTop10Corr'] = json.dumps(pp.top10corr)
            all_df_symbols.loc[indx, 'wTop10xCorr'] = json.dumps(pp.top10xcorr)

    st.session_state[ss_AllDfSym] = all_df_symbols
    df_symbols = st.session_state[ss_DfSym]
    # Update the ss_DfSym DataFrame with the latest from all_df_symbols
    for row in df_symbols.itertuples():
        if row.Symbol in all_df_symbols.Symbol.values:
            indx = df_symbols.index[df_symbols.Symbol == row.Symbol]
            df_symbols.loc[indx, 'LongName'] = row.LongName
            df_symbols.loc[indx, 'Groups'] = row.Groups
            df_symbols.loc[indx, 'Trend'] = row.Trend
            df_symbols.loc[indx, 'DlyPrdStg'] = row.DlyPrdStg
            df_symbols.loc[indx, 'WklyPrdStg'] = row.WklyPrdStg
            df_symbols.loc[indx, 'dTop10Corr'] = row.dTop10Corr
            df_symbols.loc[indx, 'dTop10xCorr'] = row.dTop10xCorr
            df_symbols.loc[indx, 'wTop10Corr'] = row.wTop10Corr
            df_symbols.loc[indx, 'wTop10xCorr'] = row.wTop10xCorr

    st.session_state[ss_DfSym] = df_symbols

    return all_df_symbols


def sym_correlations(prd, st, sym_dpps, prog_bar, just_one=None):

    # Minimum number of data points required for correlation calculations
    min_data_points = 50

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
        if just_one is not None and tsym != just_one:
            # Skip all symbols except the one specified
            continue

        target_sym = sym_dpps[tsym]
        target_sym.ticker = tsym  # Make sure the ticker is set correctly

        if target_sym.orig_data is None or len(target_sym.orig_data) < min_data_points:
            len_ = None
            if target_sym.orig_data is not None:
                len_ = len(target_sym.orig_data)
            logger.info(f"target_sym[{target_sym.ticker} {target_sym.period}] [{len_}] has less than {min_data_points} data points.")
            continue

        for ssym in sym_dpps.keys():

            if tsym != ssym:
                source_sym = sym_dpps[ssym]

                if source_sym.orig_data is None or len(source_sym.orig_data) < min_data_points:
                    len_ = None
                    if source_sym.orig_data is not None:
                        len_ = len(source_sym.orig_data)
                    logger.info(
                        f"Symbol [{source_sym.ticker}] [{len_}] has less than {min_data_points}s data points. Wont calculate correlations.")
                    continue

                corr = target_sym.periodic_correlation(source_sym)
                sym_corr[(tsym, ssym)] = corr
                # print(f"Cross Correlation for {tsym} and {ssym} completed.")
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
        corrs.append((ts, (round(sym_corr[ts]['pct_corr'], 5),
                           round(sym_corr[ts]['pct_uncorr'], 5))))
    # Sort the correlations by the pct_uncorr value
    srt_corrs = sorted(corrs, key=lambda tup: tup[1][0], reverse=True)
    srt_xcorrs = sorted(corrs, key=lambda tup: tup[1][1], reverse=True)
    # For each symbol, get the top 10 correlations
    item_cnt = len(sym_dpps)
    i = 0
    for tsym in sym_dpps.keys():
        i += 1
        # Update the progress bar
        prog_bar.progress(int(i / item_cnt * 100), f"{prd} Correlations (3): {tsym} ({i}/{item_cnt})")
        if just_one is not None and tsym != just_one:
            # Skip all symbols except the one specified
            continue
        target_sym = sym_dpps[tsym]
        sym_dpps[tsym].ticker = tsym  # Make sure the ticker is set correctly

        if target_sym.orig_data is None or len(target_sym.orig_data) < min_data_points:
            len_ = None
            if target_sym.orig_data is not None:
                len_ = len(target_sym.orig_data)
            logger.info(
                f"Symbol [{target_sym.ticker}] [{len_}] has less than {min_data_points} data points. Wont calculate correlations.")
            continue

        top10corr = []
        j = 0
        for ssym in srt_corrs:
            if j > 10:
                break
            if tsym == ssym[0][0]:
                top10corr.append((ssym[0][1], ssym[1][0]))
                j += 1
        target_sym.top10corr = top10corr

        top10xcorr = []
        j = 0
        for ssym in srt_xcorrs:
            if j > 10:
                break
            if tsym == ssym[0][0]:
                top10xcorr.append((ssym[0][1], ssym[1][1]))
                j += 1
        target_sym.top10xcorr = top10xcorr

    return


if __name__ == "__main__":
    my_msg = "I'm Still Here!"
    # asyncio.run(main(my_msg))
    main(my_msg)
