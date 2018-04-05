import pandas as pd
import multiprocessing
import time
import numpy as np
import warnings
import datetime

from tools.db_connector import save_ddr2_flights_to_db

warnings.filterwarnings("ignore")


def create_epoch_ts(r):
    try:
        dt_b = str(r['t_seg_b']) + '_' + str(r['dd_seg_b'])
        dt_e = str(r['t_seg_e']) + '_' + str(r['dd_seg_e'])

        ep_seg_b = (
                    datetime.datetime.strptime(dt_b, '%H%M%S_%y%m%d') - datetime.datetime(1970, 1, 1)).total_seconds()
        ep_seg_e = (
                    datetime.datetime.strptime(dt_e, '%H%M%S_%y%m%d') - datetime.datetime(1970, 1, 1)).total_seconds()
    except Exception as e:
        ep_seg_b = np.nan
        ep_seg_e = np.nan

    return ep_seg_b, ep_seg_e


def process_flid_df(df_in):
    df = df_in

    eb_l = []
    ee_l = []

    for i, d in df.iterrows():
        eb, ee = create_epoch_ts(d)
        eb_l.append(eb)
        ee_l.append(ee)

    df['ep_seg_b'] = eb_l
    df['ep_seg_e'] = ee_l

    df[['lat_seg_b', 'lon_seg_b', 'lat_seg_e', 'lon_seg_e']] = df[['lat_seg_b', 'lon_seg_b', 'lat_seg_e',
                                                                   'lon_seg_e']] / 60
    df = df.drop(['t_seg_b', 't_seg_e', 'dd_seg_b', 'dd_seg_e'], axis=1)

    save_ddr2_flights_to_db(df)

    return True


def parallelize_on_flid_save_to_db(df_in, func):
    """group the dataframe by icao address and process these using parallelization"""

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print("Number of unique icaos: %d", len(df_in['flight_id'].unique()))

    res = pool.map(func, [group for name, group in pd.groupby(df_in, by=['flight_id'])])
    pool.close()
    pool.join()

    return True


def parse_ddr2_file(fin, cols, ftype):

    t_start = time.time()
    df_ddr2 = pd.read_csv(fin, names=cols, delim_whitespace=True)
    df_ddr2['data_type'] = ftype
    parallelize_on_flid_save_to_db(df_ddr2, process_flid_df)

    t_pc = time.time()-t_start
    print("Parsing finished in %s seconds" % str(t_pc))


if __name__ == '__main__':

    cols = ['seg_id', 'org', 'dst', 'ac_type', 't_seg_b', 't_seg_e', 'fl_seg_b',
            'fl_seg_e', 'status', 'callsgn', 'dd_seg_b', 'dd_seg_e', 'lat_seg_b',
            'lon_seg_b', 'lat_seg_e', 'lon_seg_e', 'flight_id', 'seq', 'seg_len',
            'seg_par']

    # I/O file locations to use
    from os import listdir
    from os.path import isfile, join
    fpath = "D:\Victor\OneDrive\Documents\Studie\Msc. Thesis\Data\To Parse\DDR2"
    files = [f for f in listdir(fpath)]

    for f_csv in files:
        f_in = join(fpath, f_csv)
        ftype = f_csv.strip('.so6').split('_')[-1]
        print(f_in)
        print(ftype)
        parse_ddr2_file(f_in, cols, ftype)
