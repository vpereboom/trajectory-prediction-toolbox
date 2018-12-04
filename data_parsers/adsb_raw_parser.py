"""
Author: Victor Pereboom
Code based on script from J. Sun (TU Delft)

Decode an ADS-B dump using multiprocessing
"""

from __future__ import print_function
import pandas as pd
import pyModeS as pms
import multiprocessing
import time
import warnings

from tools.flight_segregator import *
from tools.db_connector import save_adsb_flights_to_db

warnings.filterwarnings("ignore")


def decode_airspeed(msg):
    """return velocity params from message"""
    if isinstance(msg, str):
        try:
            v = pms.adsb.velocity(msg)
            spd, hdg, roc, _ = v
        except Exception as e:
            spd, hdg, roc = None, None, None
    else:
        spd, hdg, roc = None, None, None

    return spd, hdg, roc


def decode_oe_flag(msg):
    """decode the oe flag"""
    try:
        oe = pms.adsb.oe_flag(msg)
    except Exception as e:
        oe = None
    return oe


def decode_callsign(msg):
    """decode the callsign"""
    try:
        c = pms.adsb.callsign(msg)
    except Exception as e:
        c = None
    return c


def process_pos_df(df_pos, cols):

    # Location (lat, lon) of the ADS-B receiver antenna
    sil_lat = 51.990
    sil_lon = 4.375
    lat0 = sil_lat
    lon0 = sil_lon

    icao = df_pos['icao'].unique()[0]
    df_pos.dropna(inplace=True)
    # df_pos.drop_duplicates(['ts', 'icao'], inplace=True)

    positions = []

    last_even_msg = ''
    last_odd_msg = ''
    last_even_time = 0
    last_odd_time = 0

    for i, d in df_pos.iterrows():

        oe = decode_oe_flag(d['msg'])  # identify the ODD / EVEN flag for each message

        if oe == 0:
            last_even_msg = d['msg']
            last_even_time = d['ts']
        else:
            last_odd_msg = d['msg']
            last_odd_time = d['ts']

        if abs(last_even_time - last_odd_time) < 10:
            if pms.adsb.typecode(last_even_msg) != pms.adsb.typecode(last_odd_msg):
                continue

            p = pms.adsb.position(last_even_msg, last_odd_msg,
                                  last_even_time, last_odd_time,
                                  lat0, lon0)

            if not p:
                continue

            if last_even_time > last_odd_time:
                ts = last_even_time
                try:
                    alt = pms.adsb.altitude(last_even_msg)
                except Exception as e:
                    alt = np.nan
            else:
                ts = last_odd_time
                try:
                    alt = pms.adsb.altitude(last_odd_msg)
                except Exception as e:
                    alt = np.nan

            positions.append({
                'ts': round(ts, 2),
                'ts_rounded': int(round(ts)),
                'icao': icao,
                'lat': p[0],
                'lon': p[1],
                'alt': alt
            })
        else:
            positions.append({
                'ts': round(d['ts'], 2),
                'ts_rounded': int(round(d['ts'])),
                'icao': icao,
                'lat': np.nan,
                'lon': np.nan,
                'alt': np.nan
            })

    df_pos_ret = pd.DataFrame(positions)

    return df_pos_ret[cols]


def process_spd_df(df_spd, cols):

    df_spd.dropna(inplace=True)
    df_spd['ts_rounded'] = df_spd['ts'].round().astype(int)
    # df_spd.drop_duplicates(['ts_rounded'], inplace=True)

    spd_l, hdg_l, roc_l = [], [], []
    for i, d in df_spd.iterrows():
        spd, hdg, roc = decode_airspeed(d['msg'])
        spd_l.append(spd)
        hdg_l.append(hdg)
        roc_l.append(roc)

    df_spd['spd'] = spd_l
    df_spd['hdg'] = hdg_l
    df_spd['roc'] = roc_l

    return df_spd[cols]


def process_callsign_df(df_callsign, cols):

    df_callsign['ts_rounded'] = df_callsign['ts'].round().astype(int)
    # df_callsign.drop_duplicates(['ts_rounded'], inplace=True)

    cs = []
    for i, d in df_callsign.iterrows():
        cs.append(decode_callsign(d['msg']))

    df_callsign['callsign'] = cs

    return df_callsign[cols]


def process_icao_df(df_raw, merge_on="pos", save_to_db=True):
    # Expected column names in the final data frame
    cols = ['ts', 'icao', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc', 'callsign']

    pname = multiprocessing.Process().name

    pos_cols = ['ts', 'ts_rounded', 'icao', 'lat', 'lon', 'alt']
    spd_cols = ['ts_rounded', 'spd', 'hdg', 'roc']
    callsign_cols = ['ts_rounded', 'callsign']

    # Splitting DF based on Type Codes
    df_pos_raw = df_raw[(df_raw['tc'].between(9, 18)) | (df_raw['tc'].between(5, 8))]
    df_callsign_raw = df_raw[df_raw['tc'].between(1, 4)]
    df_spd_raw = df_raw[(df_raw['tc'] == 19) | (df_raw['tc'].between(5, 8))]

    if any(ln == 0 for ln in [len(df_pos_raw), len(df_callsign_raw), len(df_spd_raw)]):
        print("DF empty")
        return pd.DataFrame(columns=cols)

    else:
        # loop through all ac messages to decode the positions
        print('%s: decoding positions...' % pname)
        df_pos_decoded = process_pos_df(df_pos_raw, pos_cols)

        # loop through all ac messages to decode the airspeed
        print("%s: decoding velocities..." % pname)
        df_spd_decoded = process_spd_df(df_spd_raw, spd_cols)

        # loop through all ac messages to decode the callsign
        print("%s: decoding callsigns..." % pname)
        df_callsign_decoded = process_callsign_df(df_callsign_raw, callsign_cols)

        # merge velocity message to decoded positions
        if merge_on == 'pos':
            merge_type = 'left'
            df_spd_decoded.drop_duplicates(['ts_rounded'], inplace=True)
        else:  # if mergeon == 'v':
            merge_type = 'right'
            df_pos_decoded.drop_duplicates(['ts_rounded'], inplace=True)

        df_merged = df_pos_decoded.merge(df_spd_decoded, on=['ts_rounded'], how=merge_type)
        df_merged = df_merged.merge(df_callsign_decoded, on=['ts_rounded'], how='left')

        # df_merged.drop_duplicates(['icao', 'lat', 'lon', 'spd', 'hdg', 'roc'], inplace=True)
        df_merged.drop_duplicates(['ts_rounded'], inplace=True)
        df_merged = df_merged[cols]

        if save_to_db:
            sep_df = separate_flights(df_merged)
            save_adsb_flights_to_db(sep_df)
            return True

        return df_merged


def parallelize_on_icao(df_in, func):
    """group the dataframe by icao address and process these using parallelization"""

    pool_cpu_size = 6 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(pool_cpu_size)
    print("Number of unique icaos: %d", len(df_in['icao'].unique()))

    df_processed = pd.concat(pool.map(func, [group for name, group in pd.groupby(df_in, by=['icao'])]))
    pool.close()
    pool.join()

    return df_processed


def parallelize_on_icao_save_to_db(df_in, func):
    """group the dataframe by icao address and process these using parallelization"""

    pool_cpu_size = 6 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(pool_cpu_size)
    print("Number of unique icaos: %d", len(df_in['icao'].unique()))

    res = pool.map(func, [group for name, group in pd.groupby(df_in, by=['icao'])])
    pool.close()
    pool.join()

    return True


def parse_adsb_file(fin):

    t_start = time.time()
    df_adsb = pd.read_csv(fin, names=['ts', 'tc', 'icao', 'msg'])
    parallelize_on_icao_save_to_db(df_adsb, process_icao_df)

    t_pc = time.time()-t_start
    print("Parsing finished in %s seconds" % str(t_pc))

    return True


# if __name__ == '__main__':
#     # I/O file locations to use
#     from os import listdir
#     from os.path import join
#     fpath = "D:\Victor\OneDrive\Documents\Studie\Msc. Thesis\Data\To Parse\ADSB"
#     files = [f for f in listdir(fpath)]
#
#     for f_csv in files:
#         f_in = join(fpath, f_csv)
#         print(f_in)
#         f_out = "output.csv"
#         parse_adsb_file(f_in, f_out)
