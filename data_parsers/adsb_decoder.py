"""
Decode an ADS-B dump using multiprocessing, run following for arguments
python decode_adsb_multi_process.py -h
Decrease CHUNKSIZE for low memory computer.
"""

from __future__ import print_function
import pandas as pd
import pyModeS as pms
import multiprocessing
import warnings
import time
import numpy as np
warnings.filterwarnings("ignore")

sil_lat = 51.990
sil_lon = 4.375

COLS = ['ts', 'icao', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc', 'callsign']

CHUNKSIZE = 1000000
N_PARTITIONS = 10

fin = "D:\Victor\OneDrive\Documents\Studie\Msc. Thesis\Data\ADSB_RAW_20171021.csv"
fout = "output_icao.csv"
mergeon = "pos"
lat0 = sil_lat  # float(args.lat0)
lon0 = sil_lon  # float(args.lon0)


def decode_airspeed(msg):
    """return velocity params from message"""
    if isinstance(msg, str):
        v = pms.adsb.velocity(msg)
        spd, hdg, roc, _ = v
    else:
        spd, hdg, roc = None, None, None

    return pd.Series({'spd': spd, 'hdg': hdg, 'roc': roc})


def add_last_msg_data(msg):
    return


def decode_position(msg):
    return


def decode_oe_flag(msg):
    return pms.adsb.oe_flag(msg)


def decode_callsign(msg):
    return pms.adsb.callsign(msg)


def add_eo_msg(r):

    [lem, let, lom, lot] = [np.nan, np.nan, np.nan, np.nan]

    if r['oe'] == 1:
        lem = r['msg']
        let = r['ts']
        lom = np.nan
        lot = np.nan
    else:
        lem = np.nan
        let = np.nan
        lom = r['msg']
        lot = r['ts']

    r['last_even_msg'] = lem
    r['last_even_time'] = let
    r['last_odd_msg'] = lom
    r['last_odd_time'] = lot
    return r


def calc_pos_output(r):

    r['ts'] = np.nan
    r['ts_rounded'] = np.nan
    r['icao'] = np.nan
    r['lat'] = np.nan
    r['lon'] = np.nan
    r['alt'] = np.nan

    if abs(r['last_even_time'] - r['last_odd_time']) < 10:
        if pms.adsb.typecode(r['last_even_msg']) != pms.adsb.typecode(r['last_odd_msg']):
            return r

        p = pms.adsb.position(r['last_even_msg'], r['last_odd_msg'],
                              r['last_even_time'], r['last_odd_time'],
                              lat0, lon0)

        if not p:
            return r

        if r['last_even_time'] > r['last_odd_time']:
            ts = r['last_even_time']
            try:
                alt = pms.adsb.altitude(r['last_even_msg'])
            except Exception as e:
                alt = np.nan
        else:
            ts = r['last_odd_time']
            try:
                alt = pms.adsb.altitude(r['last_odd_msg'])
            except Exception as e:
                alt = np.nan

        r['ts'] = round(ts, 2)
        r['ts_rounded'] = int(round(ts))
        r['icao'] = r['icao']
        r['lat'] = p[0]
        r['lon'] = p[1]
        r['alt'] = alt

        return r

    else:
        return r


# def process_pos_df(df_pos, cols):
#     # ['ts', 'ts_rounded', 'icao', 'lat', 'lon', 'alt']
#
#     icao_len = len(df_pos['icao'].unique())
#     assert icao_len == 1, "Multiple icaos in DF"
#
#     df_pos = df_pos.apply(decode_oe_flag, axis=1)    # identify the ODD / EVEN flag for each message
#     df_pos.dropna(inplace=True)
#     df_pos.drop_duplicates(['ts', 'icao'], inplace=True)
#
#     df_pos = df_pos.apply(add_eo_msg, axis=1)
#     df_pos[['last_even_msg', 'last_even_time', 'last_odd_msg', 'last_odd_time']].ffill(inplace=True)
#
#     df_pos = df_pos.apply(calc_pos_output, axis=1)
#     df_pos_ret = df_pos[cols]
#     df_pos_ret = df_pos_ret.dropna(how='all')
#
#     return df_pos_ret

def process_pos_df(df_pos, cols):

    if len(df_pos) == 0:
        return pd.DataFrame(columns=cols)

    icao_len = len(df_pos['icao'].unique())
    # print(df_pos['icao'].unique())
    assert icao_len == 1, "Multiple icaos in DF"

    icao = df_pos['icao'].unique()[0]
    df_pos['oe'] = df_pos['msg'].apply(decode_oe_flag)    # identify the ODD / EVEN flag for each message
    df_pos.dropna(inplace=True)
    df_pos.drop_duplicates(['ts', 'icao'], inplace=True)

    positions = []

    last_even_msg = ''
    last_odd_msg = ''
    last_even_time = 0
    last_odd_time = 0

    for i, d in df_pos.iterrows():
        if d['oe'] == 0:
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
                alt = pms.adsb.altitude(last_even_msg)
            else:
                ts = last_odd_time
                alt = pms.adsb.altitude(last_odd_msg)

            positions.append({
                'ts': round(ts, 2),
                'ts_rounded': int(round(ts)),
                'icao': icao,
                'lat': p[0],
                'lon': p[1],
                'alt': alt
            })
        else:
            continue

    df_pos_ret = pd.DataFrame(positions)

    return df_pos_ret[cols]


def process_spd_df(df_spd, cols):
    if len(df_spd) == 0:
        return pd.DataFrame(columns=cols)
    df_spd.dropna(inplace=True)
    df_spd['ts'] = df_spd['ts'].round(2)
    df_spd = df_spd.join(df_spd['msg'].apply(decode_airspeed))
    df_spd.drop_duplicates(['ts'], inplace=True)
    df_spd['ts_rounded'] = df_spd['ts'].round().astype(int)
    df_spd = df_spd.drop(['ts', 'icao', 'tc'], axis=1)
    # assert set(cols).issubset(df_spd.columns()), "Expected spd columns not equal"

    return df_spd


def process_callsign_df(df_callsign, cols):
    if len(df_callsign) == 0:
        return pd.DataFrame(columns=cols)
    df_callsign['ts_rounded'] = df_callsign['ts'].round().astype(int)
    df_callsign.drop_duplicates(['ts'], inplace=True)
    df_callsign['callsign'] = df_callsign['msg'].apply(decode_callsign)
    df_callsign = df_callsign.drop(['msg', 'icao', 'ts', 'tc'], axis=1)
    # assert set(cols).issubset(df_callsign.columns()), "Expected callsign columns not equal"
    try:
        df_callsign = df_callsign[cols]
    except Exception as e:
        print("Columns not equal")
        df_callsign = pd.DataFrame(columns=cols)
        pass

    return df_callsign


def process_icao_df(df_raw):
    pname = multiprocessing.Process().name

    pos_cols = ['ts', 'ts_rounded', 'icao', 'lat', 'lon', 'alt']
    spd_cols = ['ts_rounded', 'spd', 'hdg', 'roc']
    callsign_cols = ['ts_rounded', 'callsign']

    # Splitting DF based on Type Codes
    df_pos_raw = df_raw[(df_raw['tc'].between(9, 18)) | (df_raw['tc'].between(5, 8))]
    df_callsign_raw = df_raw[df_raw['tc'].between(1, 4)]
    df_spd_raw = df_raw[(df_raw['tc'] == 19) | (df_raw['tc'].between(5, 8))]

    if any(l == 0 for l in [len(df_pos_raw), len(df_callsign_raw), len(df_spd_raw)]):
        print("DF empty")
        return pd.DataFrame(columns=COLS)

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
    if mergeon == 'pos':
        merge_type = 'left'
        df_spd_decoded.drop_duplicates(['ts_rounded'], inplace=True)
    else:  # if mergeon == 'v':
        merge_type = 'right'
        df_pos_decoded.drop_duplicates(['ts_rounded'], inplace=True)

    df_merged = df_pos_decoded.merge(df_spd_decoded, on=['ts_rounded'], how=merge_type)
    # df_merged = df_merged.join(df_merged['msg'].apply(decode_airspeed))

    df_merged = df_merged.merge(df_callsign_decoded, on=['ts_rounded'], how='left')
    df_merged.drop_duplicates(['icao', 'lat', 'lon', 'spd', 'hdg', 'roc'], inplace=True)
    df_merged = df_merged[COLS]

    return df_merged


def parallelize_on_icao(df_in, func):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print("Number of unique icaos: %d", len(df_in['icao'].unique()))
    df_processed = pd.concat(pool.map(func, [group for name, group in pd.groupby(df_in, by=['icao'])]))
    pool.close()
    pool.join()

    return df_processed


if __name__ == '__main__':
    start_time = time.time()
    df_adsb = pd.read_csv(fin, names=['ts', 'icao', 'tc', 'msg'])

    df_out = parallelize_on_icao(df_adsb, process_icao_df)
    df_out.sort_values(['icao', 'ts'], inplace=True)

    print("Write to csv file: %s, %d lines\n" % (fout, df_out.shape[0]))
    df_out.to_csv(fout, index=False, header=False)
    print("--- %s seconds ---" % (time.time() - start_time))
