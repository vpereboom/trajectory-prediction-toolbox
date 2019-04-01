import pandas as pd
import multiprocessing
import time
import numpy as np
import warnings
import datetime

from tools.db_connector import save_to_pg
from conf.config import input_data_columns

warnings.filterwarnings("ignore")

cols = ["flight_length", "data_type", "flight_id", "org", "dst",
                     "start_ep", "end_ep", "ac_type", "callsign", "seg_id",
                     "ep_seg_b", "ep_seg_e", "fl_seg_b", "fl_seg_e", "status",
                     "lat_seg_b", "lon_seg_b", "lat_seg_e", "lon_seg_e",
                     "seq", "seg_len", "seg_par", "start_lat", "start_lon",
                     "end_lat", "end_lon"]


def create_epoch_ts(ts):
    try:

        ep = (datetime.datetime.strptime(str(int(ts)), '%Y%m%d%H%M%S') -
              datetime.datetime(1970, 1, 1)).total_seconds()

        return ep

    except Exception as e:
        print(e)
        return np.nan


def get_latlon(crd):

    c = crd
    lat = int(c[:2]) + (int(c[2:4]) * 60 + int(c[4:6])) / 3600
    lon = int(c[7:10]) + (int(c[10:12])*60 + int(c[12:14]))/3600

    if c[14] == 'W':
        lon = -lon

    if c[6] == 'S':
        lat = -lat

    return lat, lon


def process_allft_row(r):

    r_dict = {}

    try:
        r_dict['flight_length'] = r['flight_length']
        r_dict['data_type'] = 'm3'
        r_dict['org'] = r['org']
        r_dict['dst'] = r['dst']
        r_dict['ac_type'] = r['ac_type']
        r_dict['callsign'] = r['icao']

        r_dict['seg_id'] = []
        r_dict['ep_seg_b'] = []
        r_dict['ep_seg_e'] = []
        r_dict['fl_seg_b'] = []
        r_dict['fl_seg_e'] = []
        r_dict['status'] = []
        r_dict['lat_seg_b'] = []
        r_dict['lon_seg_b'] = []
        r_dict['lat_seg_e'] = []
        r_dict['lon_seg_e'] = []
        r_dict['seq'] = []
        r_dict['seg_len'] = []
        r_dict['seg_par'] = []

        fl_df = pd.DataFrame([i.split(':') for i in r['fl_points'].split(' ')])
        fl_df.columns = ['t_seg_b', 'seg_id', 't_seg_e', 'unknown',
                         'seg_crd_b', 'seg_crd_e', 'fl_seg_b', 'fl_seg_e',
                         't_elapsed_b', 't_elapsed_e']

        if len(fl_df) == 0:
            return None

        for i, d in fl_df.iterrows():
            r_dict['seg_id'].append(d['seg_id'])
            r_dict['ep_seg_b'].append(create_epoch_ts(d['t_seg_b']))
            r_dict['ep_seg_e'].append(create_epoch_ts(d['t_seg_e']))
            r_dict['fl_seg_b'].append(d['fl_seg_b'])
            r_dict['fl_seg_e'].append(d['fl_seg_b'])
            r_dict['status'].append(0)

            lat_b, lon_b = get_latlon(d['seg_crd_b'])
            lat_e, lon_e = get_latlon(d['seg_crd_e'])

            r_dict['lat_seg_b'].append(lat_b)
            r_dict['lon_seg_b'].append(lon_b)
            r_dict['lat_seg_e'].append(lat_e)
            r_dict['lon_seg_e'].append(lon_e)
            r_dict['seq'].append(i)
            r_dict['seg_len'].append(0)
            r_dict['seg_par'].append(0)

        r_dict['start_ep'] = r_dict['ep_seg_b'][0]
        r_dict['end_ep'] = r_dict['ep_seg_e'][-1]
        r_dict['start_lat'] = r_dict['lat_seg_b'][0]
        r_dict['start_lon'] = r_dict['lon_seg_b'][0]
        r_dict['end_lat'] = r_dict['lat_seg_e'][-1]
        r_dict['end_lon'] = r_dict['lon_seg_e'][-1]
        r_dict['flight_id'] = '%s_%s' % (str(r['icao']), str(r_dict['start_ep']))

    except Exception as e:
        print(e)
        return False

    try:
        save_to_pg([r_dict], 'allft_flights')
    except Exception as e:
        print(e)
        return False

    return True


def process_ftfm_row(r):
    r_dict = {}

    try:
        r_dict['flight_length'] = r['flight_length_as']
        r_dict['data_type'] = 'm3'
        r_dict['org'] = r['org']
        r_dict['dst'] = r['dst']
        r_dict['ac_type'] = r['ac_type']
        r_dict['callsign'] = r['icao']

        r_dict['seg_id'] = []
        r_dict['ep_seg_b'] = []
        r_dict['ep_seg_e'] = []
        r_dict['fl_seg_b'] = []
        r_dict['fl_seg_e'] = []
        r_dict['status'] = []
        r_dict['lat_seg_b'] = []
        r_dict['lon_seg_b'] = []
        r_dict['lat_seg_e'] = []
        r_dict['lon_seg_e'] = []
        r_dict['seq'] = []
        r_dict['seg_len'] = []
        r_dict['seg_par'] = []

        fl_df = pd.DataFrame(
            [i.split(':') for i in r['fl_points_as'].split(' ')])
        fl_df.columns = ['t_seg_b', 'seg_id', 't_seg_e', 'unknown',
                         'seg_crd_b', 'seg_crd_e', 'fl_seg_b', 'fl_seg_e',
                         't_elapsed_b', 't_elapsed_e']

        if len(fl_df) == 0:
            return False

        for i, d in fl_df.iterrows():
            r_dict['seg_id'].append(d['seg_id'])
            r_dict['ep_seg_b'].append(create_epoch_ts(d['t_seg_b']))
            r_dict['ep_seg_e'].append(create_epoch_ts(d['t_seg_e']))
            r_dict['fl_seg_b'].append(d['fl_seg_b'])
            r_dict['fl_seg_e'].append(d['fl_seg_e'])
            r_dict['status'].append(0)

            lat_b, lon_b = get_latlon(d['seg_crd_b'])
            lat_e, lon_e = get_latlon(d['seg_crd_e'])

            r_dict['lat_seg_b'].append(lat_b)
            r_dict['lon_seg_b'].append(lon_b)
            r_dict['lat_seg_e'].append(lat_e)
            r_dict['lon_seg_e'].append(lon_e)
            r_dict['seq'].append(i)
            r_dict['seg_len'].append(0)
            r_dict['seg_par'].append(0)

        r_dict['start_ep'] = r_dict['ep_seg_b'][0]
        r_dict['end_ep'] = r_dict['ep_seg_e'][-1]
        r_dict['start_lat'] = r_dict['lat_seg_b'][0]
        r_dict['start_lon'] = r_dict['lon_seg_b'][0]
        r_dict['end_lat'] = r_dict['lat_seg_e'][-1]
        r_dict['end_lon'] = r_dict['lon_seg_e'][-1]
        r_dict['flight_id'] = '%s_%s' % (
        str(r['icao']), str(r_dict['start_ep']))

    except Exception as e:
        print(e)
        return False

    try:
        save_to_pg([r_dict], 'allft_flights')
    except Exception as e:
        print(e)
        return False

    return True


def process_ctfm_row(r):
    r_dict = {}

    try:
        r_dict['flight_length'] = r['flight_length']
        r_dict['org'] = r['org']
        r_dict['dst'] = r['dst']
        r_dict['ac_type'] = r['ac_type']
        r_dict['callsign'] = r['icao']
        r_dict['data_type'] = 'ctfm'

        r_dict['seg_id'] = []
        r_dict['ep_seg_b'] = []
        r_dict['ep_seg_e'] = []
        r_dict['fl_seg_b'] = []
        r_dict['fl_seg_e'] = []
        r_dict['status'] = []
        r_dict['lat_seg_b'] = []
        r_dict['lon_seg_b'] = []
        r_dict['lat_seg_e'] = []
        r_dict['lon_seg_e'] = []
        r_dict['seq'] = []
        r_dict['seg_len'] = []
        r_dict['seg_par'] = []

        fl_df = pd.DataFrame(
            [i.split(':') for i in r['fl_points'].split(' ')])
        fl_df.columns = ['ts', 'wp', 'loc_type', 'fl', 'time_el',
                         'msg_type', 'crd', 'unknown_var', 'unknown_status']

        for i, d in fl_df.iterrows():

            r_dict['seg_id'].append(d['wp'])
            r_dict['ep_seg_b'].append(create_epoch_ts(d['ts']))
            r_dict['fl_seg_b'].append(d['fl'])
            r_dict['status'].append(0)

            lat, lon = get_latlon(d['crd'])

            r_dict['lat_seg_b'].append(lat)
            r_dict['lon_seg_b'].append(lon)

            r_dict['seq'].append(i)
            r_dict['seg_len'].append(0)
            r_dict['seg_par'].append(0)

        r_dict['ep_seg_e'] = r_dict['ep_seg_b'][1:]
        r_dict['fl_seg_e'] = r_dict['fl_seg_b'][1:]
        r_dict['lon_seg_e'] = r_dict['lon_seg_b'][1:]
        r_dict['lat_seg_e'] = r_dict['lat_seg_b'][1:]

        for k in ['seg_id', 'ep_seg_b', 'fl_seg_b', 'status', 'lat_seg_b',
                  'lon_seg_b', 'seq', 'seg_len', 'seg_par']:
            r_dict[k] = r_dict[k][:-1]

        r_dict['start_ep'] = r_dict['ep_seg_b'][0]
        r_dict['end_ep'] = r_dict['ep_seg_e'][-1]
        r_dict['start_lat'] = r_dict['lat_seg_b'][0]
        r_dict['start_lon'] = r_dict['lon_seg_b'][0]
        r_dict['end_lat'] = r_dict['lat_seg_e'][-1]
        r_dict['end_lon'] = r_dict['lon_seg_e'][-1]
        r_dict['flight_id'] = '%s_%s' % (
        str(r['icao']), str(r_dict['start_ep']))

    except Exception as e:
        print(e)
        return False

    try:
        save_to_pg([r_dict], 'ctfm_flights')
    except Exception as e:
        print(e)
        return False

    return True


if __name__ == '__main__':

    cols = input_data_columns['ddr2']

    # I/O file locations to use
    from os import listdir
    from os.path import isfile, join
    fpath = "/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/to_parse/allft"
    files = [f for f in listdir(fpath)]

    for f_csv in files:
        f_in = join(fpath, f_csv)
        print(f_in)

        t_start = time.time()
        df_ddr2 = pd.read_csv(f_in, names=[str(i) for i in range(172)],
                              sep=';', usecols=[str(ii) for ii in
                                                [0, 1, 2, 4, 5, 7,
                                                 57, 112, 113]],
                              compression='gzip')

        df_ddr2.columns = ['org', 'dst', 'icao', 'ac_type', 'start_t', 'end_t',
                           'callsign', 'flight_length', 'fl_points']

        df_ddr2 = df_ddr2.dropna()

        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        print("Number of unique icaos: %d", len(df_ddr2))
        rlst = [r for i, r in df_ddr2.iterrows()]
        res = pool.map(process_ctfm_row, rlst)
        pool.close()
        pool.join()
        print("Parsing finished in %s seconds" % str(time.time() - t_start))

