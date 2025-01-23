"""
File: Correlation Testing

Observations:
- The combination of 2 symbols with the high cointegration and stationarity (a Pairs Trade Potential or PPP) at the same
  time depends on the period length. And as the period length increases, pairs trade potential decreases and PPPs
  come in and out of existence.
- A true PPP would also require a review via a Spread_Mean analysis in which it show the spread between the two
  going out 2 standard deviations from the mean and then coming back in, several times. And then testing in simulation
  with a profit. And showing a quick trading cycle in/out.     
- It may also be possible to find a PPP with a longer opposing trend, and then only trading the pair on days where
  the trade requirements match each sock's longer trend.

Parallelization:
- Push PPOs into a berkeley db.
- Create env set up script for other nodes.
- Set Script package and object DB to send to other nodes.
- Send Script and Data packages to other nodes.
- Run env setup script on other nodes.
- Find available cores on other nodes.
- Allocate 80% of cores to the script.
- Determine total available resources.
- Run scripts that process 1 symbol against all other symbols (per core).
  - Collect results from nodes as they finish.
  - Send new work the nodes as they finish.

Additional Optimization:
    - Save out all (symbol_a vs symbol_b) correlations that have already been evaluated.
    - Before running a correlation test, check if the correlation has already been evaluated.
    - Check for (symbol_a vs symbol_b) and (symbol_b vs symbol_a) in the saved correlations.
    - Skip the correlation test if the correlation has already been evaluated.
    - Save these correlation to the

Notes:
     - Tried concurrent.futures, ThreadPoolExecutor, ProcessPoolExecutor, and multiprocessing.Pool but all had issues with
       locking up, and the lockup was not possible to track down. This seems to lockup just at dill.loads().
     - After reading the article: https://pythonspeed.com/articles/python-multiprocessing/
       - Why your multiprocessing Pool is stuck (it’s full of sharks!)
       using multiprocessing.Pool() and set_start_method('spawn', force=True), that seems to do that trick.
       The issue has do to with the artifacts of forking. But by using 'spawn' an execve() is done after the form()
       which forces the child to be en entirely new process so that no state is inherited from the parent.
"""
import os
import pandas as pd
import berkeleydb as bdb
import multiprocessing as mp

from functools import partial
from multiprocessing import set_start_method, get_context
from tqdm import tqdm
from pricepredict import PricePredict
from datetime import datetime, timedelta

set_start_method('spawn', force=True)

model_dir = './models/'
chart_dir = './charts/'
preds_dir = './predictions/'
ppo_dir = './ppo/'
ppo_save_dir = './ppo_save/'

global DbEnv, ObjCache
DbEnv = None
ObjCache = None


def open_obj_cache_write(clear_db=False):
    db_env_dir = 'ppo/cache'
    db_file = 'ppo_cache.db'
    db_filepath = f'{db_env_dir}/{db_file}'

    if clear_db:
        # Remove all the Berkeley database filex in the db_env_dir
        print(f'Removing all files in {db_env_dir}')
        for file in os.listdir(db_env_dir):
            print(f' - Removing {file}')
            os.remove(f'{db_env_dir}/{file}')

    # Make sure that the permissions on the db file are 666
    if os.path.exists(db_filepath):
        if oct(os.stat(db_filepath).st_mode)[-3:] != '666':
            os.chmod(db_filepath, 0o666)

    # Creates a Berkeley DB environment and a Berkeley DB object cache that can be used my multiple processes.
    db_env = bdb.db.DBEnv()
    env_flags = bdb.db.DB_CREATE | bdb.db.DB_INIT_LOCK | bdb.db.DB_INIT_TXN | bdb.db.DB_INIT_MPOOL | bdb.db.DB_THREAD
    db_env.open(db_env_dir, env_flags, 0)
    bdb_flags = bdb.db.DB_CREATE
    obj_cache = bdb.db.DB(db_env)
    obj_cache.open(db_file, dbtype=bdb.db.DB_BTREE, flags=bdb_flags)

    # print("ObjCache Opened Read/Write")
    return obj_cache


def open_obj_cache_readonly():

    # Creates a Berkeley DB environment and a Berkeley DB object cache that can be used my multiple processes.
    DbEnv = bdb.db.DBEnv()
    env_flags = bdb.db.DB_INIT_MPOOL | bdb.db.DB_INIT_TXN | bdb.db.DB_INIT_MPOOL | bdb.db.DB_THREAD
    DbEnv.open('ppo/cache', env_flags, 0)
    bdb_flags = bdb.db.DB_RDONLY
    ObjCache = bdb.db.DB(DbEnv)
    ObjCache.open('ppo_cache.db', dbtype=bdb.db.DB_BTREE, flags=bdb_flags)

    # print("ObjCache Opened Read-Only")
    return ObjCache


def read_daily_ppos(symbols: [str] = None, setup_db=False) -> dict:
    """
    Read the PPO objects from the ppo_dir
    """

    obj_cache = None

    if setup_db:
        # Open the Object Cache
        obj_cache = open_obj_cache_write(clear_db=True)
    else:
        obj_cache = open_obj_cache_readonly()


    ret_ppos = {}

    if len(obj_cache) > 50:
        # print(f"... ObjCache Length: {len(obj_cache)}\n{obj_cache.keys()}")
        # Get PP Object from Object Cache
        pb_objcache = tqdm(total=len(obj_cache), desc=f'Reading [{len(obj_cache)}] PPOs from Object Cache')
        for sym in obj_cache.keys():
            pb_objcache.update(1)
            if symbols is not None:
                if str(sym) not in symbols:
                    continue
            if sym.startswith(b'||'):
                continue
            else:
                ppo = PricePredict.unserialize(obj_cache[sym])
                ret_ppos[ppo.ticker] = ppo
                # print(f" - Loaded PPO for {sym} --> {ppo.ticker}")

    elif len(obj_cache) == 0:

        # Open the Object Cache in read-only mode
        obj_cache.close()
        obj_cache = open_obj_cache_write()

        # Read Daily PPOs from the ppo_dir
        # Count the number of files that contain _D_ in the name in the ppo_dir
        f_count = len([f for f in os.listdir(ppo_dir) if '_D_' in f and f.endswith('.dilz')])
        pb_files = tqdm(total=f_count, desc=f'Reading [{f_count}] PPOs from File System')
        for file in os.listdir(ppo_dir):
            # Check if filename has _D_ in it and ends with .dill
            if '_D_' in file and file.endswith('.dilz'):
                # Get the symbol name from the file name (first chars before _D_)
                (sym, prd, odt) = file.split('_')
                if symbols is not None:
                    if sym not in symbols:
                        continue
                # Load the PPO object from the file
                with open(f'{ppo_dir}/{file}', 'rb') as f:
                    b_obj = f.read()
                    # # Store the compressed PPO object in the Object Cache
                    obj_cache[bytes(sym + '_' + prd, 'latin1')] = b_obj
                    # unserialize the PPO object, using the current Class for PricePredict
                    ppo = PricePredict.unserialize(b_obj, ignore=False)
                    ret_ppos[ppo.ticker] = ppo
                pb_files.update(1)

    if setup_db:
        # Commit the Object Cache
        obj_cache.sync()

    # Close the Object Cache
    obj_cache.close()

    return ret_ppos


def create_ppos(symbols: [str]):
    # Create a PricePredict object for each symbol
    ppos = {}
    for sym in symbols:
        ppo = PricePredict(sym, period=PricePredict.PeriodDaily,
                           model_dir=model_dir,
                           chart_dir=chart_dir,
                           preds_dir=preds_dir,)
        end_dt = datetime.now()
        # Load up over 5 years of data
        start_dt = end_dt - timedelta(days=365 * 5)
        end_date = end_dt.strftime('%Y-%m-%d')
        start_date = start_dt.strftime('%Y-%m-%d')

        # Fetch data for the ppo
        try:
            ppo.fetch_data_yahoo(ppo.ticker, start_date, end_date)
        except Exception as e:
            print(f'Error fetching data for {sym}')
            continue

        ppos[sym] = ppo

    return ppos


def clear_obj_cache_pairs():
    obj_cache = open_obj_cache_write()
    for key in obj_cache.keys():
        if key.startswith(b'||'):
            del obj_cache[key]
    obj_cache.sync()
    obj_cache.close()


def check_pair_evaluated(obj_cache, symbol1, symbol2, pc_period):
    # Check if the correlation between symbol1 and symbol2 has already been evaluated
    if obj_cache.get(bytes('||' + symbol1 + ':' + symbol2 + ':' + str(pc_period), 'latin1')) is not None:
        return True
    elif obj_cache.get(bytes('||' + symbol2 + ':' + symbol1 + ':' + str(pc_period), 'latin1')) is not None:
        return True
    else:
        obj_cache[bytes('||' + symbol1 + ':' + symbol2 + ':' + str(pc_period), 'latin1')] = b'Evaluated'
    return False


def gen_pair_trading_analysis(symbol1, mp_lock=None, return_dict=None):
    """
    This function is our main worker function that will be run in parallel by the Pool.map() function.
    It needs to read all of the daily PPO objects from the cache, and then run a correlation test between
    symbol1 and all other symbols to find historical cointegration and stationarity between the pairs.
    It will open the object cache in read-only mode, and then close it when done.
    """
    sym_ppos = read_daily_ppos()

    obj_cache = open_obj_cache_write()

    # print(f'Loaded {len(sym_ppos)} daily PPO objects')
    # print(f'Daily Symbols: {sym_ppos.keys()}')

    ppo_symbols = sorted(sym_ppos.keys())
    all_symbols = []
    for sym in ppo_symbols:
        if sym.startswith('||'):
            continue
        else:
            all_symbols.append(sym)

    # print(f'Loaded {len(sym_ppos)} daily PPO objects')
    all_ptps = None

    # From 7 days ago to 300 days ago, in steps of 2 days
    # We run a correlation test between each combinatorial pair of symbols
    # to find historical cointegration and stationarity between the pairs.
    # We only consider pairs that are cointegrated and stationary.
    start_period = 7
    end_period = 300
    step_days = 2
    # Create count for periods to test.
    periods = range(start_period, end_period, step_days)
    # Create a progress bar for the periods
    mp_lock.acquire() if mp_lock is not None else None
    pb_periods = tqdm(total=len(periods), desc=f'Testing [{symbol1}] vs all other symbols')
    mp_lock.release() if mp_lock is not None else None
    for pc_period in range(start_period, end_period, step_days):
        mp_lock.acquire() if mp_lock is not None else None
        pb_periods.update(1)
        mp_lock.release() if mp_lock is not None else None
        for symbol2 in all_symbols:
            if symbol1 != symbol2:
                # Get the corr between the two symbols
                try:
                    # Check if symbol1 vs symbol2 has already been evaluated
                    if check_pair_evaluated(obj_cache, symbol1, symbol2, pc_period):
                        continue
                    ppo1 = sym_ppos[symbol1]
                    ppo2 = sym_ppos[symbol2]
                    corr = ppo1.periodic_correlation(ppo2, period_len=pc_period)
                except Exception as e:
                    print(f'Error calculating correlation between {symbol1} and {symbol2} period: [{pc_period}]\n'
                          + f'  - Exception: {e}')
                    continue

                if corr['coint_stationary']:
                    corr_dict = {'potential_pair': f'{symbol1}:{symbol2}',
                                 'corr_start_date': corr['start_date'], 'corr_end_date': corr['end_date'],
                                 'period_days': corr['corr_period_len'],
                                 'coint_stasn': corr['coint_stationary'],
                                 'coint_pval': corr['coint_test']['p_val'],
                                 'adf_pval': corr['adf_test']['p_val']}
                    ptp = pd.DataFrame(corr_dict, index=[0])
                    if all_ptps is None:
                        all_ptps = ptp
                    else:
                        all_ptps = pd.concat([all_ptps, ptp])

    obj_cache.sync()
    obj_cache.close()

    if return_dict is not None:
        return_dict[symbol1] = all_ptps

    return all_ptps, symbol1


# ====================== main ========================

if __name__ == '__main__':
    # Read and cache the daily PPO objects
    sym_ppos = read_daily_ppos(setup_db=False)
    # sym_ppos = read_daily_ppos()

    clear_obj_cache_pairs()

    # Parallelize the correlation testing over 12 cores using pultiprocessing
    num_processes = 4
    app_trading_pairs = None
    tasks_submitted = len(sym_ppos)
    tasks_returned = 0
    with get_context('spawn').Pool(processes=num_processes) as pool:
        processes = []
        queue = mp.Queue()
        manager = mp.Manager()
        return_dict = manager.dict()
        lock = manager.Lock()
        # The partial function is used to pass the mp_lock and return_dict to the
        # gen_pair_trading_analysis function as these parameters cannot be passed directly
        # to the pool.map() function.
        partial_worker = partial(gen_pair_trading_analysis, mp_lock=None, return_dict=return_dict)
        # Hand a symbol to the gen_pair_trading_analysis function via concurrent.futures.
        results = pool.map(partial_worker, sorted(sym_ppos.keys()))

        for task_id in return_dict.keys():
            all_ptp = return_dict[task_id]
            if app_trading_pairs is None:
                app_trading_pairs = all_ptp
            else:
                app_trading_pairs = pd.concat([app_trading_pairs, all_ptp])
            tasks_returned += 1

    all_trade_pairs = app_trading_pairs.sort_values(['potential_pair', 'period_days'])
    all_trade_pairs.to_csv('all_trade_pairs.csv', index=False)
    print(f"All Tasks Submitted: [{tasks_submitted}], All Tasks Returned: [{tasks_returned}]")
    print(f'All Trade Pairs: [{len(all_trade_pairs)}]')

