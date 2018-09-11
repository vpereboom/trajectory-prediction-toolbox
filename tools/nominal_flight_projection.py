import pandas as pd
import numpy as np
import math
from pymongo import MongoClient
import psycopg2 as psql
from psycopg2.extras import RealDictCursor
import time

cl = MongoClient()
db = cl['flights']
col = db['adsb_flights']

# TODO Implement metric conventions


def find_coord_dst_hdg(coord1, hdg, dst):
    # https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing

    R = 6378.1  # Radius of the Earth in km
    hdg = math.radians(hdg)  # Bearing is converted to radians.
    d = dst / 1000  # Distance in km

    lat1 = math.radians(coord1[0])  # Current lat point converted to radians
    lon1 = math.radians(coord1[1])  # Current long point converted to radians

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
        math.cos(lat1) * math.sin(d / R) * math.cos(hdg))

    y = math.sin(hdg) * math.sin(d / R) * math.cos(lat1)
    x = math.cos(d / R) - math.sin(lat1) * math.sin(lat2)

    lon2 = lon1 + math.atan2(y, x)

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lat2, lon2)


def nominal_proj(fl_df, look_ahead_t):
    proj_coord_lat = [np.nan]*len(fl_df)
    proj_coord_lon = [np.nan]*len(fl_df)

    r = {}
    nr = 0
    for k in fl_df.keys():
        r[k] = nr
        nr = nr+1

    for i, row in enumerate(fl_df.values):
        if i == 0:
            coord_start = (row[r['lat']], row[r['lon']])
            hdg_start = row[r['hdg']]
            spd_start = row[r['spd']]
            ts_start = row[r['ts']]
        else:
            if (row[r['ts']] - ts_start) < look_ahead_t:
                dst_start = (row[r['ts']] - ts_start) * (spd_start * 0.514444)  # TODO Put fixed values in global or config
                crd = find_coord_dst_hdg(coord_start, hdg_start, dst_start)
                proj_coord_lat[i] = crd[0]
                proj_coord_lon[i] = crd[1]
            else:
                break

    fl_df['proj_lat'] = proj_coord_lat
    fl_df['proj_lon'] = proj_coord_lon

    return fl_df


def nominal_proj_avg(fl_df, look_ahead_t=600, hdg_start_nr=5):
    proj_coord_lat = []
    proj_coord_lon = []

    coord_start = (fl_df['lat'].iloc[hdg_start_nr], fl_df['lon'].iloc[hdg_start_nr])
    hdg_avg_start = fl_df['hdg'].iloc[list(range(0, hdg_start_nr))].mean()
    spd_avg_start = fl_df['spd'].iloc[list(range(0, hdg_start_nr))].mean()
    ts_start = fl_df['ts'].iloc[hdg_start_nr]
    fl_df['proj_lat'] = np.nan
    fl_df['proj_lon'] = np.nan

    for i in range(hdg_start_nr, len(fl_df)):
        if (fl_df['ts'].iloc[i] - ts_start) < look_ahead_t:
            dst_start = (fl_df['ts'].iloc[i] - ts_start) * (spd_avg_start * 0.514444)
            crd = find_coord_dst_hdg(coord_start, hdg_avg_start, dst_start)
            proj_coord_lat.extend([crd[0]])
            proj_coord_lon.extend([crd[1]])
        else:
            proj_coord_lat.extend([np.nan])
            proj_coord_lon.extend([np.nan])

    fl_df['proj_lat'].iloc[range(hdg_start_nr, len(fl_df))] = proj_coord_lat
    fl_df['proj_lon'].iloc[range(hdg_start_nr, len(fl_df))] = proj_coord_lon

    return fl_df


def calc_coord_dst(c1, c2):
    R = 6371.1 * 1000  # Radius of the Earth in m

    lat1 = c1[0]
    lon1 = c1[1]
    lat2 = c2[0]
    lon2 = c2[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d


def calc_coord_dst_simple(c1, c2):
    R = 6371.1 * 1000  # Radius of the Earth in m

    lon1 = c1[0]
    lat1 = c1[1]
    lon2 = c2[0]
    lat2 = c2[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    x = dlon * math.cos(dlat / 2)
    y = dlat
    d = math.sqrt(x * x + y * y) * R

    return d


def calc_bearing(c0, c1):
    if not all(isinstance(i, tuple) for i in [c0, c1]):
        return np.nan

    lat1 = c0[0]
    lon1 = c0[1]
    lat2 = c1[0]
    lon2 = c1[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1

    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    y = math.sin(dlon) * math.cos(lat2)
    bearing = math.atan2(y, x)

    return math.degrees(bearing)


def calc_compass_bearing(c0, c1):
    if not all(isinstance(i, tuple) for i in [c0, c1]):
        return np.nan

    lat1 = c0[0]
    lon1 = c0[1]
    lat2 = c1[0]
    lon2 = c1[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1

    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    y = math.sin(dlon) * math.cos(lat2)
    bearing = math.atan2(y, x)

    return (math.degrees(bearing) + 360) % 360


def get_triangle_corner(d0, d1, d2):
    bb = (d0 ** 2 + d1 ** 2 - d2 ** 2) / (2 * d0 * d1)
    return math.acos(bb)


def convert_knts_ms(spd):
    return spd*0.51444


def calc_track_errors(wp_last, wp_curr, wp_proj):
    assert all(isinstance(i, tuple) for i in [wp_last, wp_curr, wp_proj]), "Coordinate entries are not tuples"

    dst_last_curr = calc_coord_dst_simple(wp_last, wp_curr)
    dst_last_proj = calc_coord_dst_simple(wp_last, wp_proj)
    dst_proj_curr = calc_coord_dst_simple(wp_curr, wp_proj)

    hdg_wp0_wpproj = calc_bearing(wp_last, wp_proj)
    hdg_wp0_wpac = calc_bearing(wp_last, wp_curr)

    alpha_2 = get_triangle_corner(dst_last_curr, dst_last_proj, dst_proj_curr)

    if hdg_wp0_wpproj - hdg_wp0_wpac < 0:
        alpha_2 = -1*alpha_2

    cte = math.sin(alpha_2) * dst_last_curr
    tte = dst_proj_curr
    ate = math.sqrt(tte ** 2 - cte ** 2)

    if dst_last_curr < dst_last_proj:
        ate = -1*ate

    return cte, ate, tte


def create_projection_dict(fl_dd, lookahead_t):

    fl_dd = fl_dd[fl_dd['hdg'].first_valid_index():]
    fl_dd = fl_dd.reset_index(drop=True)

    fl_dd = fl_dd.iloc[fl_dd.first_valid_index():]

    if len(fl_dd) == 0:
        return None

    proj_df = nominal_proj(fl_dd, lookahead_t)

    cte_arr = []
    ate_arr = []
    tte_arr = []
    proj_df['prev_lat'] = proj_df['lat'].shift(1)
    proj_df['prev_lon'] = proj_df['lon'].shift(1)

    r = {}
    nr = 0
    for k in proj_df.keys():
        r[k] = nr
        nr = nr + 1

    for ii, row in enumerate(proj_df.values):
        if ii != 0:
            wp_last = (row[r['prev_lat']], row[r['prev_lon']])
            wp_curr = (row[r['lat']], row[r['lon']])
            wp_proj = (row[r['proj_lat']], row[r['proj_lon']])
            cte, ate, tte = calc_track_errors(wp_last, wp_curr, wp_proj)
            cte_arr.append(cte)
            ate_arr.append(ate)
            tte_arr.append(tte)
        else:
            cte_arr.append(0)
            ate_arr.append(0)
            tte_arr.append(0)

    proj_df['cte'] = cte_arr
    proj_df['ate'] = ate_arr
    proj_df['tte'] = tte_arr
    proj_df['time_proj'] = proj_df['time_el'] - proj_df['time_el'].min()

    # proj_df = proj_df.drop(columns=['_id'])
    dct = proj_df.to_dict(orient="list")
    for k in ["flight_length", "icao", "flight_id", "alt_min", "alt_max", "start_lat", "start_lon", "end_lat", "end_lon", "start_ep", "end_ep", "callsign", "flight_number"]:
        dct[k] = dct[k][0]

    return dct


def flush_to_db(insert_lst, conn):
    try:
        records_list_template = ','.join(['%s'] * len(insert_lst))
        insert_query = 'insert into projected_flights (flight_length, icao, flight_id, alt_min, alt_max, \
                    start_lat, start_lon, end_lat, end_lon, start_ep, end_ep, callsign, flight_number, ts, lat, lon, \
                    alt, spd, hdg, roc, time_el, proj_lat, proj_lon, prev_lat, prev_lon, cte, ate, tte, time_proj) \
                    values {}'.format(records_list_template)

        cur_insert = conn.cursor()
        sql_inj_list = [tuple([d[k] for k in column_order_lst_proj]) for d in insert_lst]
        cur_insert.execute(insert_query, sql_inj_list)
        conn.commit()
        cur_insert.close()

        return True

    except Exception as e:
        print("Flush to DB failed:")
        print(e)

        return False


if __name__ == "__main__":

    la_time = 900
    cnt = 0
    cnt_max = np.inf
    fl_len_min = la_time*1.2
    flight_level_min = 20000

    column_order_lst_proj = ["flight_length", "icao", "flight_id", "alt_min", "alt_max", "start_lat", "start_lon", "end_lat",
                        "end_lon", "start_ep", "end_ep", "callsign", "flight_number", "ts", "lat", "lon", "alt", "spd",
                        "hdg", "roc", "time_el", "proj_lat", "proj_lon", "prev_lat", "prev_lon", "cte", "ate", "tte",
                        "time_proj"]

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("I am unable to connect to the database.")
        print(e)

    max_inserts = 100
    fetch_batch_size = max_inserts
    cnt = 0

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT * FROM public.adsb_flights WHERE flight_length > %s;", (fl_len_min,))

    while True:

        batch = cur_read.fetchmany(size=fetch_batch_size)

        if not batch:
            break

        insert_lst = []
        for fl in batch:

            t0 = time.time()

            fl_d = pd.DataFrame.from_dict(fl)
            fl_d = fl_d[fl_d['alt'] > flight_level_min]
            fl_d['time_el'] = fl_d['ts'] - fl_d['ts'].min()

            # Clean up the dataframe by patching NAN values
            fl_d['hdg'] = fl_d['hdg'].fillna(method='ffill')
            fl_d['spd'] = fl_d['spd'].fillna(method='ffill')

            if fl_d['time_el'].max() < la_time:
                print("flight too short")
                continue

            try:
                steps = int(fl_d['time_el'].max() / la_time)

                for i in range(steps):
                    dct = create_projection_dict(fl_d[(fl_d['time_el'] > i * la_time) & (fl_d['time_el'] < (i+1) * la_time)], la_time)
                    if dct:
                        insert_lst.append(dct)
                        cnt = cnt + 1

            except Exception as e:
                print(e)
                continue

            print("Flight %d took %f sec" % (cnt, time.time() - t0))

        # Flush to DB
        flush_stat = flush_to_db(insert_lst, conn)
        if flush_stat:
            print("%d Flights inserted" % len(insert_lst))

    print("Done")
    cur_read.close()