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
from conf.config import input_data_columns, sil_lat, sil_lon, \
                        segmentation_window, cpu_count, flight_df_minlen

warnings.filterwarnings("ignore")


def decode_airspeed(msg):
    """return velocity params from message"""

    if not msg:
        return None, None, None

    if isinstance(msg, str):
        try:
            v = pms.adsb.velocity(msg)
            if not v:
                return None, None, None
            spd, hdg, roc, _ = v
            return spd, hdg, roc
        except Exception as e:
            print('Decoding Airspeed failed')
            print(e)
            return None, None, None
    else:
        return None, None, None


def decode_oe_flag(msg):
    """decode the oe flag"""

    if not msg:
        return None, None, None

    try:
        return pms.adsb.oe_flag(msg)
    except Exception as e:
        print('Decoding oe Flag failed')
        print(e)
        return None


def decode_callsign(msg):
    """decode the callsign"""

    if not msg:
        return None, None, None

    try:
        return pms.adsb.callsign(msg)
    except Exception as e:
        print('Decoding Callsign failed')
        print(e)
        return None


def process_pos_df(df_pos, cols):

    # Location (lat, lon) of the ADS-B receiver antenna
    lat0 = sil_lat
    lon0 = sil_lon

    icao = df_pos['icao'].unique()[0]
    df_pos.dropna(inplace=True)

    positions = []

    last_even_msg = ''
    last_odd_msg = ''
    last_even_time = 0
    last_odd_time = 0

    for i, d in df_pos.iterrows():

        if not d['msg']:
            continue

        # identify the ODD / EVEN flag for each message
        try:
            oe = decode_oe_flag(d['msg'])
        except Exception as e:
            print('Error in decoding oe flag')
            print(e)
            continue

        if oe is None:
            continue

        if oe == 0:
            last_even_msg = d['msg']
            last_even_time = d['ts']
        else:
            last_odd_msg = d['msg']
            last_odd_time = d['ts']

        if abs(last_even_time - last_odd_time) < 10:
            if pms.adsb.typecode(last_even_msg) != \
                    pms.adsb.typecode(last_odd_msg):
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
                    print('Error in decoding altitude')
                    print(e)
                    alt = np.nan

            else:
                ts = last_odd_time
                try:
                    alt = pms.adsb.altitude(last_odd_msg)

                except Exception as e:
                    print('Error in decoding altitude')
                    print(e)
                    alt = np.nan

            positions.append({
                'ts': ts,
                'ts_int': int(ts),
                'icao': icao,
                'lat': p[0],
                'lon': p[1],
                'alt': alt
            })
        else:
            positions.append({
                'ts': d['ts'],
                'ts_int': int(d['ts']),
                'icao': icao,
                'lat': np.nan,
                'lon': np.nan,
                'alt': np.nan
            })

    df_pos_ret = pd.DataFrame(positions)
    df_pos_ret = df_pos_ret.dropna(subset=['ts', 'lat', 'lon'])

    return df_pos_ret[cols]


def process_spd_df(df_spd, cols):

    df_spd.dropna(inplace=True)

    spd_l, hdg_l, roc_l, ts_l = [], [], [], []

    for i, d in df_spd.iterrows():

        if not d['msg']:
            continue

        try:
            spd, hdg, roc = decode_airspeed(d['msg'])
        except Exception as e:
            print('Error in decoding airspeed')
            print(e)
            continue

        if not spd:
            continue

        spd_l.append(spd)
        hdg_l.append(hdg)
        roc_l.append(roc)
        ts_l.append(int(d['ts']))

    df_spd_new = pd.DataFrame()
    df_spd_new['spd'] = spd_l
    df_spd_new['hdg'] = hdg_l
    df_spd_new['roc'] = roc_l
    df_spd_new['ts_int'] = ts_l

    df_spd_new = df_spd_new.fillna(method='ffill', limit=5)
    df_spd_new.dropna(inplace=True)

    return df_spd_new[cols]


def process_callsign_df(df_callsign, cols):

    # df_callsign['ts_int'] = df_callsign['ts'].astype(int)
    # df_callsign.drop_duplicates(['ts_rounded'], inplace=True)

    cs = []
    ts = []

    for i, d in df_callsign.iterrows():

        if not d['msg']:
            continue

        try:
            csi = decode_callsign(d['msg'])
        except Exception as e:
            print('Error in decoding altitude')
            print(e)
            continue

        if not csi:
            continue

        cs.append(csi)
        ts.append(int(d['ts']))

    df_callsign_new = pd.DataFrame()
    df_callsign_new['callsign'] = cs
    df_callsign_new['ts_int'] = ts

    return df_callsign_new[cols]


def process_icao_df(df_raw, merge_on="pos", save_to_db=True):
    # Expected column names in the final data frame
    cols = ['ts', 'icao', 'lat', 'lon', 'alt',
            'spd', 'hdg', 'roc', 'callsign']

    pos_cols = ['ts', 'ts_int', 'icao', 'lat', 'lon', 'alt']
    spd_cols = ['ts_int', 'spd', 'hdg', 'roc']
    callsign_cols = ['ts_int', 'callsign']

    # Splitting DF based on Type Codes
    df_pos_raw = df_raw[(df_raw['tc'].between(9, 18)) |
                        (df_raw['tc'].between(5, 8))]

    df_callsign_raw = df_raw[df_raw['tc'].between(1, 4)]

    df_spd_raw = df_raw[(df_raw['tc'] == 19) |
                        (df_raw['tc'].between(5, 8))]

    for ln in [(len(df_pos_raw), 'POS'), (len(df_callsign_raw), 'CALLSGN'),
               (len(df_spd_raw), 'ŚPD')]:
        if ln[0] == 0:
            print("%s df empty" % ln[1])
            return False

    # loop through all ac messages to decode the positions
    try:
        df_pos_decoded = process_pos_df(df_pos_raw, pos_cols)
    except Exception as e:
        print('Parsing position df failed')
        print(e)
        return False

    # loop through all ac messages to decode the airspeed
    try:
        df_spd_decoded = process_spd_df(df_spd_raw, spd_cols)
    except Exception as e:
        print('Parsing airspeed df failed')
        print(e)
        return False

    # loop through all ac messages to decode the callsign
    try:
        df_callsign_decoded = process_callsign_df(df_callsign_raw,
                                                  callsign_cols)

        callsigns = [c for c in
                     df_callsign_decoded['callsign'].dropna().unique()
                     if c is not None]

    except Exception as e:
        print('Parsing callsign df failed')
        print(e)
        return False

    # merge velocity message to decoded positions

    try:
        df_merged = pd.merge_asof(df_pos_decoded, df_spd_decoded,
                                  on='ts_int', direction='nearest',
                                  tolerance=5)

        df_merged = pd.merge_asof(df_merged, df_callsign_decoded,
                                  on='ts_int')

        df_merged.drop_duplicates(subset=['ts', 'icao', 'lat', 'lon'],
                                  inplace=True)

        df_merged = df_merged.fillna(method='ffill', limit=5)
        df_merged = df_merged.dropna()
        df_merged = df_merged[cols]

    except Exception as e:
        print('Merging dataframes failed')
        print(e)
        return False

    if save_to_db:
        if len(df_merged) < flight_df_minlen:
            # print('Flight not saved to db: DF has too few data points')
            return False

        try:
            sep_df = separate_flights(df_merged,
                                      split_window=segmentation_window)
            save_adsb_flights_to_db(sep_df, coll='adsb_flights_v2')
            return True

        except Exception as e:
            print('Segregating and Saving to database failed')
            print(e)
            return False

    return df_merged


def parallelize_on_icao(df_in, func):
    """group the dataframe by icao address and process \
    these using parallelization"""

    pool_cpu_size = cpu_count
    pool = multiprocessing.Pool(pool_cpu_size)
    print("Number of unique icaos: %d", len(df_in['icao'].unique()))

    df_processed = pd.concat(pool.map(func, [group for name, group
                                             in pd.groupby(df_in,
                                                           by=['icao'])]))
    pool.close()
    pool.join()

    return df_processed


def parallelize_on_icao_save_to_db(df_in, func):
    """group the dataframe by icao address and process \
    these using parallelization"""

    pool_cpu_size = cpu_count
    pool = multiprocessing.Pool(pool_cpu_size)
    print("Number of unique icaos: %d", len(df_in['icao'].unique()))

    res = pool.map(func, [group for name, group
                          in pd.groupby(df_in, by=['icao'])])
    pool.close()
    pool.join()

    return True


def parse_adsb_file(fin):

    t_start = time.time()

    cols = input_data_columns['adsb']
    cols_old = input_data_columns['adsb_old']

    if 'RAW' in fin:
        df_adsb = pd.read_csv(fin, header=None, names=cols_old)

    else:
        df_adsb = pd.read_csv(fin, header=0, names=cols)
        df_adsb = df_adsb[['ts', 'tc', 'icao', 'msg']]

    df_adsb = df_adsb.drop_duplicates()
    parallelize_on_icao_save_to_db(df_adsb, process_icao_df)

    t_pc = time.time()-t_start
    print("Parsing finished in %s seconds" % str(t_pc))

    return True
