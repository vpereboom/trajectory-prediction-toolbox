import numpy as np
import math
import time
import psycopg2 as psql
import itertools
import datetime
import multiprocessing
import pandas as pd
from psycopg2.extras import RealDictCursor


def create_bounding_box(f):
    fuz = list(zip(*f))

    return [np.nanmin(fuz[0]), np.nanmax(fuz[0]), np.nanmin(fuz[1]), np.nanmax(fuz[1])]


def find_box_overlap(b1, b2):
    if (b1[0] <= b2[1]) & (b1[1] <= b2[0]) or (b1[2] <= b2[3]) & (b1[3] <= b2[2]):
        return None

    else:
        return [max([b1[0], b2[0]]), min([b1[1], b2[1]]), max([b1[2], b2[2]]), min([b1[3], b2[3]])]


def resample_flight(box, f):
    """Flights should be zipped list like zip(lat,lon)"""

    f_res = [(lat, lon, ts) for lat, lon, ts in f if (box[0] <= lat <= box[1]) & (box[2] <= lon <= box[3])]

    return f_res


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


def box_area(box):
    w = calc_coord_dst_simple([box[0], box[2]], [box[1], box[2]])
    h = calc_coord_dst_simple([box[0], box[2]], [box[0], box[3]])

    return w * h


def closest_distance(f1, f2):
    """Flights should be zipped list like zip(lat,lon)"""

    x = [[np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) for c1 in f1] for c2 in f2]
    dmin = np.nanmin(x)
    if2, if1 = np.where(x == dmin)

    c2 = f2[if2[0]]
    c1 = f1[if1[0]]
    dmin_m = calc_coord_dst(c1, c2)

    return dmin_m, c1, c2, if1[0], if2[0]


def dst_aligned_times(f1, f2):
    """Flights should be zipped list like zip(lat,lon,ts)"""

    t = [[abs(c1[2] - c2[2]) for c1 in f1] for c2 in f2]
    tmin = np.nanmin(t)
    if2, if1 = np.where(t == tmin)


def find_flight_intersect(f1, f2):
    cnd = True
    first_it = True
    ba_old = None
    area_cnd = 0.95

    while cnd:
        f1_box = create_bounding_box(f1)
        f2_box = create_bounding_box(f2)

        obox = find_box_overlap(f1_box, f2_box)

        if obox:

            if not first_it:
                ba_old = ba
                ba = box_area(obox)
            else:
                ba = box_area(obox)

            if ba_old:
                if ba / ba_old > area_cnd:
                    return f1, f2

                else:

                    f1 = resample_flight(obox, f1)
                    f2 = resample_flight(obox, f2)
            else:

                f1 = resample_flight(obox, f1)
                f2 = resample_flight(obox, f2)

        else:
            return None, None

        if len(f1) == 0 or len(f2) == 0:
            return None, None

        first_it = False

        if len(f1) < 20:
            cnd = False

        # else:
        #     return None, None

    return f1, f2


def calc_coord_dst_pp(lon1, lat1, lon2, lat2):
    R = 6371.1 * 1000  # Radius of the Earth in m

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d


def postprocess_conflict(b):
    df1 = pd.DataFrame()
    for k in ['ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1']:
        df1[k.strip('_1')] = b[k]
    df2 = pd.DataFrame()
    for k in ['ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']:
        df2[k.strip('_2')] = b[k]

    df1['ts'] = df1['ts'].astype(int)
    df2['ts'] = df2['ts'].astype(int)

    if df1['ts'].max() > df2['ts'].max():
        df_l = df1
        df_r = df2
        df_r['ts_n'] = df_r['ts'].astype(int)
        sfx = ('_1', '_2')
    else:
        df_l = df2
        df_r = df1
        df_r['ts_n'] = df_r['ts'].astype(int)
        sfx = ('_2', '_1')

    dfm = pd.merge_asof(df_l, df_r, on='ts', direction='nearest', tolerance=10, suffixes=sfx)

    dfm['td'] = dfm['ts'] - dfm['ts_n']
    dfm = dfm.dropna(how='any')
    if len(dfm) > 0:
        dfm['dstd'] = dfm.apply(lambda r: calc_coord_dst_pp(r['lat_1'], r['lon_1'], r['lat_2'], r['lon_2']), axis=1)
        dst = dfm['dstd'].iloc[-1]
        dfm['ts_2'] = dfm['ts_n']
        dfm['ts_1'] = dfm['ts']

        b = dfm[['ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1',
                 'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']].to_dict(orient='list')

        b['td'] = dfm['td'].iloc[-1]
        b['altd'] = abs(dfm['alt_1'].iloc[-1] - dfm['alt_2'].iloc[-1])
        b['hdgd'] = abs(dfm['hdg_1'].iloc[-1] - dfm['hdg_2'].iloc[-1])
        b['dstd'] = dst

        return b

    else:
        return None


def get_conflicts(ep):

    fl_start_ep = ep
    ts_offset = 3600
    max_dst = 5 * 1852
    alt_min = 15000

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("Unable to connect to the database.")
        print(e)

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT ts, lat, lon, alt, spd, hdg, roc, start_ep, flight_id \
                     FROM public.adsb_flights WHERE flight_length > 500 \
                     AND start_ep BETWEEN %s AND %s;",
                     (fl_start_ep - ts_offset, fl_start_ep + ts_offset))

    f_list = []
    fsets = []
    sql_inj_lst = []
    col_lst = ['td', 'altd', 'dstd', 'hdgd',
               'flight_id_1', 'ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1',
               'flight_id_2', 'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']

    batch = cur_read.fetchall()

    id_lst = [b['flight_id'] for b in batch]
    fset = list(itertools.combinations(id_lst, 2))

    cnt = 1
    print(len(fset))

    for fs in fset:

        f1 = next(f for f in batch if f['flight_id'] == fs[0])
        f2 = next(f for f in batch if f['flight_id'] == fs[1])

        if abs(f1['start_ep'] - f2['start_ep']) < 1800:

            f1crd = [(lt, ln, ts) for lt, ln, alt, ts in
                     list(zip(f1['lat'], f1['lon'], f1['alt'], f1['ts'])) if alt > alt_min]

            f2crd = [(lt, ln, ts) for lt, ln, alt, ts in
                     list(zip(f2['lat'], f2['lon'], f2['alt'], f2['ts'])) if alt > alt_min]

            if len(f2crd) > 20 and len(f1crd) > 20:

                t1 = time.time()

                fi1, fi2 = find_flight_intersect(f1crd, f2crd)

                if fi1 and fi2:
                    d, c1, c2, i1, i2 = closest_distance(list(fi1), list(fi2))
                    bdiff = abs(calc_bearing((fi1[i1 - 1][0], fi1[i1 - 1][1]), (fi1[i1][0], fi1[i1][1])) -
                                calc_bearing((fi2[i2 - 1][0], fi2[i2 - 1][1]), (fi2[i2][0], fi2[i2][1])))
                    tdiff = abs(fi1[i1][2] - fi2[i2][2])

                    if d < max_dst and bdiff > 10:  # and tdiff < max_ts

                        f1ix = f1['ts'].index(fi1[i1][2])
                        f2ix = f2['ts'].index(fi2[i2][2])

                        confl = {}
                        for k in ['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc']:
                            confl[('%s_1' % k)] = f1[k][:f1ix]
                        for k in ['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc']:
                            confl[('%s_2' % k)] = f2[k][:f2ix]
                        # confl['flight_id_1'] = f1['flight_id']
                        # confl['flight_id_2'] = f2['flight_id']
                        # # cidstr = "%s-%s" % (f1['flight_id'], f2['flight_id'])
                        # # confl['ćid'] = cidstr
                        # confl['td'] = tdiff
                        # confl['altd'] = abs(f1['alt'][f1ix] - f2['alt'][f2ix])
                        # confl['dstd'] = d
                        # confl['hdgd'] = abs(f1['hdg'][f1ix] - f2['hdg'][f2ix])
                        #
                        # try:
                        #     sql_inj_lst.append(tuple(confl[kk] for kk in col_lst))
                        # except Exception as e1:
                        #     print(e1)

                        confln = postprocess_conflict(confl)

                        if confln:

                            confln['flight_id_1'] = f1['flight_id']
                            confln['flight_id_2'] = f2['flight_id']
                            # cidstr = "%s-%s" % (f1['flight_id'], f2['flight_id'])
                            # confl['ćid'] = cidstr
                            # confl['td'] = tdiff
                            # confl['altd'] = abs(f1['alt'][i1] - f2['alt'][i2])
                            # confl['dstd'] = d
                            # confl['hdgd'] = abs(f1['hdg'][i1] - f2['hdg'][i2])

                            if confln['dstd'] < max_dst:

                                try:
                                    sql_inj_lst.append(tuple(confln[kk] for kk in col_lst))
                                except Exception as e1:
                                    print(e1)

        cnt = cnt + 1

        if len(sql_inj_lst) > 100:
            cur_inj = conn.cursor()
            records_list_template = ','.join(['%s'] * len(sql_inj_lst))
            insert_query = 'insert into conflicts (td, altd, dstd, hdgd,\
                                                    flight_id_1, ts_1, lat_1, lon_1, alt_1, spd_1, hdg_1, roc_1,\
                                                    flight_id_2, ts_2, lat_2, lon_2, alt_2, spd_2, hdg_2, roc_2) \
                                                    values {}'.format(records_list_template)
            cur_inj.execute(insert_query, sql_inj_lst)
            conn.commit()
            cur_inj.close()
            sql_inj_lst = []

    cur_read.close()

    cur_inj = conn.cursor()
    records_list_template = ','.join(['%s'] * len(sql_inj_lst))
    insert_query = 'insert into conflicts (td, altd, dstd, hdgd,\
                                            flight_id_1, ts_1, lat_1, lon_1, alt_1, spd_1, hdg_1, roc_1,\
                                            flight_id_2, ts_2, lat_2, lon_2, alt_2, spd_2, hdg_2, roc_2) \
                                            values {}'.format(records_list_template)
    cur_inj.execute(insert_query, sql_inj_lst)
    conn.commit()
    cur_inj.close()
    sql_inj_lst = []

    conn.close()


if __name__ == "__main__":

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("Unable to connect to the database.")
        print(e)

    cur_ts = conn.cursor(cursor_factory=RealDictCursor)
    cur_ts.execute("SELECT start_ep, flight_id \
                     FROM public.adsb_flights;")

    batch = cur_ts.fetchall()
    x = [b['start_ep'] for b in batch]
    daylst = list(set([datetime.datetime.fromtimestamp(xx).strftime('%Y-%m-%d %H') for xx in x]))
    eplist = [(datetime.datetime.strptime(ts, '%Y-%m-%d %H') - datetime.datetime(1970, 1, 1)).total_seconds() for ts in
              daylst]

    pool = multiprocessing.Pool((multiprocessing.cpu_count() - 2))

    res = pool.map(get_conflicts, eplist)
    pool.close()
    pool.join()

    conn.close()
    cur_ts.close()
