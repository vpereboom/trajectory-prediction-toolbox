import numpy as np
import math
import time
import itertools
import datetime
import multiprocessing
from psycopg2.extras import RealDictCursor

from tools.flight_projection import calc_coord_dst, calc_bearing
from tools.flight_conflict_processor import postprocess_conflict
from tools.db_connector import get_pg_conn


def create_global_box(f1, f2):
    f = f1 + f2
    fuz = list(zip(*f))

    return [np.nanmin(fuz[0]), np.nanmax(fuz[0]), np.nanmin(fuz[1]), np.nanmax(fuz[1])]


def part_gl_box(box):
    latd = 0.167
    lond = 0.271

    lat_lst = [box[0] + i * latd for i in range(int(math.floor((box[1] - box[0]) / latd)))]
    lat_lst.append(box[1])
    lon_lst = [box[2] + i * lond for i in range(int(math.floor((box[3] - box[2]) / lond)))]
    lon_lst.append(box[3])

    boxes=[]

    for i in range(len(lat_lst) - 1):
        for ii in range(len(lon_lst) - 1):
            boxes.append((lat_lst[i], lat_lst[i + 1], lon_lst[ii], lon_lst[ii + 1]))

    return boxes


def check_box(f, b):
    for lat, lon, ts in f:
        if (b[0] <= lat <= b[1]) & (b[2] <= lon <= b[3]):
            return True

    return False


def find_fl_boxes(f1, f2, boxes):
    box_lst = []

    for box in boxes:
        if check_box(f1, box):
            if check_box(f2, box):
                box_lst.append(box)

    return list(set(box_lst))


def closest_distance_box(f1, f2, b):
    """Flights should be zipped list like zip(lat,lon)"""

    f1_res = [(lat, lon, ts) for lat, lon, ts in f1 if (b[0] <= lat <= b[1]) & (b[2] <= lon <= b[3])]
    f2_res = [(lat, lon, ts) for lat, lon, ts in f2 if (b[0] <= lat <= b[1]) & (b[2] <= lon <= b[3])]

    x = [[np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) for c1 in f1_res] for c2 in f2_res]
    dmin = np.nanmin(x)
    if2, if1 = np.where(x == dmin)

    c2 = f2_res[if2[0]]
    c1 = f1_res[if1[0]]
    dmin_m = calc_coord_dst(c1, c2)

    ifg1 = [i for i, x in enumerate(f1) if x[2] == f1_res[if1[0]][2]][0]
    ifg2 = [i for i, x in enumerate(f2) if x[2] == f2_res[if2[0]][2]][0]

    return dmin_m, c1, c2, ifg1, ifg2


def dst_aligned_times(f1, f2):
    """Flights should be zipped list like zip(lat,lon,ts)"""

    t = [[abs(c1[2] - c2[2]) for c1 in f1] for c2 in f2]
    tmin = np.nanmin(t)
    if2, if1 = np.where(t == tmin)


def find_flight_intersect(f1, f2):
    gl_box = create_global_box(f1, f2)
    box_lst = part_gl_box(gl_box)
    fl_boxes = find_fl_boxes(f1, f2, box_lst)

    confl_list = []

    if not fl_boxes:
        return None

    for b in fl_boxes:

        dminr, c1r, c2r, if1r, if2r = closest_distance_box(f1, f2, b)
        confl_list.append((dminr, c1r, c2r, if1r, if2r))

    return confl_list


def calc_coord_dst_pp(lon1, lat1, lon2, lat2):
    R = 6371.1 * 1000  # Radius of the Earth in m

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d


def get_conflicts(ep):

    fl_start_ep = ep
    ts_offset = 1800
    max_dst = 5 * 1852
    alt_min = 15000

    conn = get_pg_conn()

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT ts, lat, lon, alt, spd, hdg, roc, start_ep, flight_id \
                     FROM public.adsb_flights WHERE flight_length > 500 \
                     AND start_ep BETWEEN %s AND %s;",
                     (fl_start_ep - ts_offset, fl_start_ep + ts_offset))

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

                confl_list = find_flight_intersect(f1crd, f2crd)

                if confl_list:

                    for con in confl_list:

                        d, c1, c2, i1, i2 = con[0], con[1], con[2], con[3], con[4]

                        bdiff = abs(calc_bearing((f1crd[i1 - 1][0], f1crd[i1 - 1][1]), (f1crd[i1][0], f1crd[i1][1])) -
                                    calc_bearing((f2crd[i2 - 1][0], f2crd[i2 - 1][1]), (f2crd[i2][0], f2crd[i2][1])))
                        tdiff = abs(f1crd[i1][2] - f2crd[i2][2])

                        if d < max_dst and bdiff > 10:  # and tdiff < max_ts

                            confl = {}
                            for k in ['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc']:
                                confl[('%s_1' % k)] = f1[k][:i1]
                            for k in ['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc']:
                                confl[('%s_2' % k)] = f2[k][:i2]

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

                                if confln['dstd'][-1] < max_dst:

                                    try:
                                        sql_inj_lst.append(tuple(confln[kk] for kk in col_lst))
                                    except Exception as e1:
                                        print(e1)

        cnt = cnt + 1

        if len(sql_inj_lst) > 100:
            cur_inj = conn.cursor()
            records_list_template = ','.join(['%s'] * len(sql_inj_lst))
            insert_query = 'insert into conflicts_3 (td, altd, dstd, hdgd,\
                                                    flight_id_1, ts_1, lat_1, lon_1, alt_1, spd_1, hdg_1, roc_1,\
                                                    flight_id_2, ts_2, lat_2, lon_2, alt_2, spd_2, hdg_2, roc_2) \
                                                    values {}'.format(records_list_template)
            print('100 records to be added')
            cur_inj.execute(insert_query, sql_inj_lst)
            conn.commit()
            cur_inj.close()
            sql_inj_lst = []

    cur_read.close()

    cur_inj = conn.cursor()
    records_list_template = ','.join(['%s'] * len(sql_inj_lst))
    insert_query = 'insert into conflicts_3 (td, altd, dstd, hdgd,\
                                            flight_id_1, ts_1, lat_1, lon_1, alt_1, spd_1, hdg_1, roc_1,\
                                            flight_id_2, ts_2, lat_2, lon_2, alt_2, spd_2, hdg_2, roc_2) \
                                            values {}'.format(records_list_template)
    cur_inj.execute(insert_query, sql_inj_lst)
    conn.commit()
    cur_inj.close()
    sql_inj_lst = []

    conn.close()


if __name__ == "__main__":

    conn = get_pg_conn()

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
