import numpy as np
import itertools
from operator import itemgetter
import pandas as pd
from psycopg2.extras import RealDictCursor

from tools.flight_projection import calc_coord_dst, closest_distance, \
                                    calc_coord_dst_pp, calc_compass_bearing

from tools.db_connector import get_pg_conn
from tools.conflict_handling import get_delta_dst

from conf.config import la_time, flight_level_min, ipz_range, avg_sec_msg, \
                        knts_ms_ratio, ts_tol, max_adsb_gap, \
                        adsb_gap_filter_window, max_confl_time_delta, \
                        max_t_startpoints, \
                        min_fl_dataframe_length


def create_bounding_box(f):
    fuz = list(zip(*f))

    return [np.nanmin(fuz[0]), np.nanmax(fuz[0]),
            np.nanmin(fuz[1]), np.nanmax(fuz[1])]


def find_box_overlap(b1, b2):
    if (b1[0] <= b2[1]) & (b1[1] <= b2[0]) or \
            (b1[2] <= b2[3]) & (b1[3] <= b2[2]):

        return None

    else:
        return [max([b1[0], b2[0]]), min([b1[1], b2[1]]),
                max([b1[2], b2[2]]), min([b1[3], b2[3]])]


def resample_flight(box, f):
    """Flights should be zipped list like zip(lat,lon)"""

    f_res = [(fi[0], fi[1], fi[2]) for fi in f
             if (box[0] <= fi[0] <= box[1]) & (box[2] <= fi[1] <= box[3])]

    return f_res


def box_area(box):
    w = calc_coord_dst([box[0], box[2]], [box[1], box[2]])
    h = calc_coord_dst([box[0], box[2]], [box[0], box[3]])

    return w * h


def check_path_intersection(c1, c2, h1, h2):
    alpha_1 = calc_compass_bearing(c1, c2) - h1
    alpha_2 = calc_compass_bearing(c2, c1) - h2

    if ((abs(alpha_1) + abs(alpha_2)) < 180) and (alpha_1 * alpha_2 < 0):
        return True

    else:
        return False


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / n


def filter_adsb_gaps(ts):

    gaps = np.where(running_mean(np.diff(ts),
                                 adsb_gap_filter_window) > max_adsb_gap)
    gaps = gaps[0]

    if len(gaps) > 1:
        gap_tp = [[gaps[i - 1], gaps[i], gaps[i] - gaps[i - 1]]
                  for i in range(1, len(gaps))]

        tup = max(gap_tp, key=lambda x: x[2])
        tup[0] = tup[0] + (adsb_gap_filter_window - 1)
        return tuple(tup)
    else:
        return None


def find_flight_intersect(f1, f2):
    cnd = True
    first_it = True
    ba_old = None
    area_cnd = 0.95

    while cnd:

        try:
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

                        obox_1 = [obox[0], (obox[0]+obox[1])/2,
                                  obox[2], obox[3]]

                        obox_2 = [(obox[0]+obox[1])/2, obox[1],
                                  obox[2], obox[3]]

                        f1_1 = resample_flight(obox_1, f1)
                        f2_1 = resample_flight(obox_1, f2)

                        f1_2 = resample_flight(obox_2, f1)
                        f2_2 = resample_flight(obox_2, f2)

                        obox_1_res = []
                        obox_2_res = []

                        if (len(f1_1) > 1) and (len(f2_1) > 1):
                            f1_1_box = create_bounding_box(f1_1)
                            f2_1_box = create_bounding_box(f2_1)
                            obox_1_res = find_box_overlap(f1_1_box, f2_1_box)

                        if (len(f1_2) > 1) and (len(f2_2) > 1):
                            f1_2_box = create_bounding_box(f1_2)
                            f2_2_box = create_bounding_box(f2_2)
                            obox_2_res = find_box_overlap(f1_2_box, f2_2_box)

                        if obox_1_res:
                            obox = obox_1_res
                            f1 = f1_1
                            f2 = f2_1

                        elif obox_2_res:
                            obox = obox_2_res
                            f1 = f1_2
                            f2 = f2_2

                        else:
                            print('Boxing error')
                            return None, None

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
                return f1, f2

        except Exception as e:
            print(e)
            return 'Error', 'Error'


def merge_flight_dataframes(_b):

    df1 = pd.DataFrame()
    for k in ['ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1']:
        df1[k.strip('_1')] = _b[k]
    df2 = pd.DataFrame()
    for k in ['ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']:
        df2[k.strip('_2')] = _b[k]

    df1 = df1.dropna(how='any', subset=['lat', 'lon', 'alt'])
    df2 = df2.dropna(how='any', subset=['lat', 'lon', 'alt'])

    if len(df1) == 0:
        return None, None
    if len(df2) == 0:
        return None, None

    df1['ts'] = df1['ts'].astype(int)
    df2['ts'] = df2['ts'].astype(int)

    if df1['ts'].max() > df2['ts'].max():
        df_l = df1
        df_r = df2

        df_r['ts_old'] = df_r['ts']

        sfx = ('_1', '_2')
    else:
        df_l = df2
        df_r = df1

        df_l['ts_old'] = df_l['ts']

        sfx = ('_2', '_1')

    dfm = pd.merge_asof(df_l, df_r, on='ts', direction='nearest',
                        tolerance=ts_tol, suffixes=sfx)

    dfm = dfm.drop_duplicates(['lat_1', 'lon_1'])
    dfm = dfm.drop_duplicates(['lat_2', 'lon_2'])
    dfm = dfm.dropna(how='any')
    dfm = dfm.reset_index(drop=True)

    dfm_td = dfm['ts'].max() - dfm['ts'].min()

    return dfm, dfm_td


def filter_gaps_flightset(_dfm):

    ts = list(_dfm['ts'])

    ix = filter_adsb_gaps(ts)
    if not ix:
        ix = (0, -1)

    _dfm = _dfm.iloc[ix[0]:ix[1]]
    _fdt = _dfm['ts'].max() - _dfm['ts'].min()

    return _dfm, _fdt


def align_flightset(b):

    confl_flag = False

    dfm, dfm_td = merge_flight_dataframes(b)

    if dfm is None or dfm_td is None:
        return None, None

    if len(dfm) > 0 and dfm_td > la_time:

        dfm, fdt = filter_gaps_flightset(dfm)

        if len(dfm) < min_fl_dataframe_length or fdt < la_time:
            return None, None

        dfm['ts_2'] = dfm['ts_old']
        dfm['ts_1'] = dfm['ts']

        try:
            dfm['dstd'] = dfm.apply(lambda r: calc_coord_dst_pp(r['lat_1'],
                                                                r['lon_1'],
                                                                r['lat_2'],
                                                                r['lon_2']),
                                    axis=1)
        except Exception as e:
            print(e)

        dmin = dfm['dstd'].min()
        dmin_ts = dfm['ts'][dfm['dstd'] == dfm['dstd'].min()].iloc[0]

        if dmin < ipz_range:
            dfm = dfm[dfm['ts'] <= dmin_ts]
            confl_flag = True

        dfm['altd'] = dfm['alt_1'] - dfm['alt_2']
        dfm['hdgd'] = dfm['hdg_1'] - dfm['hdg_2']
        dfm['td'] = dfm['ts_1'] - dfm['ts_2']

        b = dfm[['ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1',
                 'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2',
                 'td', 'altd', 'hdgd', 'dstd']
                ].astype(float).to_dict(orient='list')

        return b, confl_flag

    else:
        return None, None


def create_artificial_conflict(_f1, _f2, _c1, _c2):

    status = False

    conflict_t1 = [fv[2] for fv in _f1 if fv[0] == _c1[0]
                   and fv[1] == _c1[1]]

    conflict_t2 = [fv[2] for fv in _f2 if fv[0] == _c2[0]
                   and fv[1] == _c2[1]]

    if not all(i for i in [conflict_t1, conflict_t2]):
        return status, _f1, _f2

    tdiff = conflict_t1[0] - conflict_t2[0]

    if abs(tdiff) < max_confl_time_delta:
        f1_new = _f1  # [fvi for fvi in _f1 if fvi[2] <= conflict_t1[0]]
        f2_new = [(fvi[0], fvi[1], fvi[2] + tdiff, fvi[3],
                   fvi[4], fvi[5], fvi[6])
                  for fvi in _f2]  # if fvi[2] <= conflict_t2[0]]

        status = True

        return status, f1_new, f2_new

    return status, _f1, _f2


def get_hourly_flight_batch(_ep):

    fl_start_ep = _ep
    ts_offset = 1800

    conn_read = get_pg_conn()
    cur_read = conn_read.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT ts, lat, lon, alt, spd, hdg, roc, start_ep, \
                         flight_id \
                         FROM public.adsb_flights \
                         WHERE (end_ep - start_ep) > %s \
                         AND ((end_ep - start_ep) / flight_length) < %s \
                         AND start_ep BETWEEN %s AND %s;",
                     (la_time, avg_sec_msg, fl_start_ep - ts_offset,
                      fl_start_ep + ts_offset))

    batch = cur_read.fetchall()

    cur_read.close()
    conn_read.close()

    return batch


def resample_flight_set(_f1, _f2, _alt_min):
    ts_upper = min(max(_f1['ts']), max(_f2['ts']))
    ts_lower = max(min(_f1['ts']), min(_f2['ts']))

    if ts_upper - ts_lower >= la_time:

        f1crd = [(lt, ln, ts, hdg, alt, roc, spd)
                 for lt, ln, ts, hdg, alt, roc, spd
                 in list(zip(_f1['lat'], _f1['lon'], _f1['ts'],
                             _f1['hdg'], _f1['alt'], _f1['roc'], _f1['spd']))
                 if alt > _alt_min and ts_lower <= ts <= ts_upper
                 and not np.isnan(hdg)]

        f2crd = [(lt, ln, ts, hdg, alt, roc, spd)
                 for lt, ln, ts, hdg, alt, roc, spd
                 in list(zip(_f2['lat'], _f2['lon'], _f2['ts'],
                             _f2['hdg'], _f2['alt'], _f2['roc'], _f2['spd']))
                 if alt > _alt_min and ts_lower <= ts <= ts_upper
                 and not np.isnan(hdg)]

        if f1crd and f2crd:
            f1td = max(f1crd, key=itemgetter(2))[2] - \
                   min(f1crd, key=itemgetter(2))[2]

            f2td = max(f2crd, key=itemgetter(2))[2] - \
                min(f2crd, key=itemgetter(2))[2]

            return f1crd, f2crd, f1td, f2td

    return None, None, None, None


def create_flightset_dict(f1_df, f2_df):
    confl = {}

    for k in ['ts', 'lat', 'lon', 'alt',
              'spd', 'hdg', 'roc']:
        confl[('%s_1' % k)] = f1_df[k].tolist()

    for k in ['ts', 'lat', 'lon', 'alt',
              'spd', 'hdg', 'roc']:
        confl[('%s_2' % k)] = f2_df[k].tolist()

    return confl


def check_flight_overlap(f1crd, f2crd):

    start_dst = calc_coord_dst((f1crd[0][0], f1crd[0][1]),
                               (f2crd[0][0], f2crd[0][1]))

    if start_dst < (f1crd[0][6] * knts_ms_ratio * max_t_startpoints):
        return True

    else:
        return False


def check_flight_convergence(f1crd, f2crd):
    # confl_flag = check_path_intersection((f1crd[0][0], f1crd[0][1]),
    #                                      (f2crd[0][0], f2crd[0][1]),
    #                                      f1crd[0][3], f2crd[0][3])

    conv_i = get_delta_dst(f1crd[0][0], f1crd[0][1], f2crd[0][0], f2crd[0][1],
                           f1crd[0][3], f2crd[0][3], f1crd[0][6], f2crd[0][6])

    if np.isnan(conv_i):
        return False
    else:
        return True


def check_conflict(f1crd, f2crd):

    fi1, fi2 = find_flight_intersect(f1crd, f2crd)

    if fi1 and fi2 and fi1 != 'Error' and fi2 != 'Error':

        d, c1, c2 = closest_distance(list(fi1), list(fi2))

        if d < ipz_range:
            return True, c1, c2
        else:
            return False, None, None

    else:
        return False, None, None


def classify_flight_pairs(ep, ctype='all'):

    assert ctype in ['all', 'conflict', 'expected', 'overlapping'], \
        "Classification type not recognized"

    alt_min = flight_level_min
    sql_inj_conflicts = []
    sql_inj_overlap = []
    sql_inj_expected = []

    col_lst = ['td', 'altd', 'dstd', 'hdgd', 'flight_id_1', 'ts_1', 'lat_1',
               'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1', 'flight_id_2',
               'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']

    batch = get_hourly_flight_batch(ep)

    id_lst = [b['flight_id'] for b in batch]
    fset = list(itertools.combinations(id_lst, 2))

    fset_len = len(fset)
    print('Set length: %d' % fset_len)

    cnt = 0

    for fs in fset:

        overlap_found, conflict_found, conv_found = False, False, False
        cnt = cnt + 1

        f1 = [f for f in batch if f['flight_id'] == fs[0]][0]
        f2 = [f for f in batch if f['flight_id'] == fs[1]][0]

        f1crd, f2crd, f1td, f2td = resample_flight_set(f1, f2, alt_min)

        if not all(i for i in [f1crd, f2crd, f1td, f2td]):
            continue

        if f1td > la_time and f2td > la_time:

            if check_flight_overlap(f1crd, f2crd):
                overlap_found = True

                _cflag, c1, c2 = check_conflict(f1crd, f2crd)
                if _cflag:
                    _cstat, f1crd, f2crd = create_artificial_conflict(f1crd,
                                                                      f2crd,
                                                                      c1, c2)
                    if _cstat:
                        conflict_found = True

                if check_flight_convergence(f1crd, f2crd):
                    conv_found = True

            if not any(c for c in [overlap_found, conflict_found,
                                   conv_found]):
                continue

            f1_df = pd.DataFrame.from_records(f1crd,
                                              columns=['lat', 'lon', 'ts',
                                                       'hdg', 'alt',
                                                       'roc', 'spd']
                                              )

            f2_df = pd.DataFrame.from_records(f2crd,
                                              columns=['lat', 'lon', 'ts',
                                                       'hdg', 'alt',
                                                       'roc', 'spd']
                                              )

            fset_dict = create_flightset_dict(f1_df, f2_df)

            try:
                fset_aligned, confl_flag = align_flightset(fset_dict)
            except Exception as e:
                print(e)
                continue

            if not fset_aligned:
                continue

            fset_aligned['flight_id_1'] = f1['flight_id']
            fset_aligned['flight_id_2'] = f2['flight_id']

            if overlap_found and not confl_flag:
                sql_inj_overlap.append(fset_aligned)

            if confl_flag:
                sql_inj_conflicts.append(fset_aligned)

            if conv_found and not confl_flag:
                sql_inj_expected.append(fset_aligned)

        else:
            continue

    if sql_inj_conflicts:

        conn = get_pg_conn()
        column_lst = col_lst

        sql_inj_list_c = [tuple([d[k] for k in column_lst])
                          for d in sql_inj_conflicts]

        records_list_template = ','.join(['%s'] * len(sql_inj_list_c))

        insert_query_c = 'insert into {} {} values {}'.format(
            'conflicts', tuple(column_lst),
            records_list_template).replace("'", "")

        try:
            cur = conn.cursor()
            cur.execute(insert_query_c, sql_inj_list_c)
            cur.close()
            conn.commit()
            conn.close()

        except Exception as e:
            print('Saving to DB failed, error: ')
            print(e)
            conn.close()

        print("%d Conflicts inserted" % len(sql_inj_conflicts))

    if sql_inj_expected:

        conn = get_pg_conn()
        column_lst = col_lst

        sql_inj_list_e = [tuple([d[k] for k in column_lst])
                          for d in sql_inj_expected]

        records_list_template = ','.join(['%s'] * len(sql_inj_list_e))

        insert_query_e = 'insert into {} {} values {}'.format(
            'converging_flights', tuple(column_lst),
            records_list_template).replace("'", "")

        try:
            cur = conn.cursor()
            cur.execute(insert_query_e, sql_inj_list_e)
            cur.close()
            conn.commit()
            conn.close()

        except Exception as e:
            print('Saving to DB failed, error: ')
            print(e)
            conn.close()

        print("%d Converging Flights inserted" % len(sql_inj_expected))

    if sql_inj_overlap:

        conn = get_pg_conn()
        column_lst = col_lst

        sql_inj_list_o = [tuple([d[k] for k in column_lst])
                          for d in sql_inj_overlap]

        records_list_template = ','.join(['%s'] * len(sql_inj_list_o))

        insert_query_o = 'insert into {} {} values {}'.format(
            'overlapping_flights', tuple(column_lst),
            records_list_template).replace("'", "")

        try:
            cur = conn.cursor()
            cur.execute(insert_query_o, sql_inj_list_o)
            cur.close()
            conn.commit()
            conn.close()

        except Exception as e:
            print('Saving to DB failed, error: ')
            print(e)
            conn.close()

        print("%d Overlapping flights inserted" % len(sql_inj_overlap))

    if not any(i for i in [sql_inj_overlap, sql_inj_expected,
                           sql_inj_conflicts]):
        print('No flight sets found')


# def get_conflicts(ep):
#
#     alt_min = flight_level_min
#     sql_inj_lst = []
#     col_lst = ['td', 'altd', 'dstd', 'hdgd', 'flight_id_1', 'ts_1', 'lat_1',
#                'lon_1', 'alt_1', 'spd_1', 'hdg_1', 'roc_1', 'flight_id_2',
#                'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2']
#
#     batch = get_hourly_flight_batch(ep)
#
#     id_lst = [b['flight_id'] for b in batch]
#     fset = list(itertools.combinations(id_lst, 2))
#
#     fset_len = len(fset)
#     print('Set length: %d' % fset_len)
#
#     cnt = 0
#
#     for fs in fset:
#         cnt = cnt + 1
#
#         f1 = [f for f in batch if f['flight_id'] == fs[0]][0]
#         f2 = [f for f in batch if f['flight_id'] == fs[1]][0]
#
#         f1crd, f2crd, f1td, f2td = resample_flight_set(f1, f2, alt_min)
#
#         if not all(i for i in [f1crd, f2crd, f1td, f2td]):
#             continue
#
#         if f1td > la_time and f2td > la_time:
#
#             fi1, fi2 = find_flight_intersect(f1crd, f2crd)
#
#             if fi1 and fi2 and fi1 != 'Error' and fi2 != 'Error':
#
#                 d, c1, c2 = closest_distance(list(fi1), list(fi2))
#
#                 if d < ipz_range:
#
#                     f1_df = pd.DataFrame.from_dict(f1)
#                     f2_df = pd.DataFrame.from_dict(f2)
#
#                     f1_ts_confl = f1_df['ts'][(f1_df['lat'] == c1[0]) &
#                                               (f1_df['lon'] == c1[1])
#                                               ].values[0]
#
#                     f2_ts_confl = f2_df['ts'][(f2_df['lat'] == c2[0]) &
#                                               (f2_df['lon'] == c2[1])
#                                               ].values[0]
#
#                     if abs(f1_ts_confl - f2_ts_confl) < \
#                             max_confl_time_delta:
#
#                         f1_df = f1_df[f1_df['ts'] <= f1_ts_confl]
#                         f2_df = f2_df[f2_df['ts'] <= f2_ts_confl]
#
#                         confl = {}
#
#                         for k in ['ts', 'lat', 'lon', 'alt',
#                                   'spd', 'hdg', 'roc']:
#
#                             confl[('%s_1' % k)] = f1_df[k].values
#
#                         for k in ['ts', 'lat', 'lon', 'alt',
#                                   'spd', 'hdg', 'roc']:
#
#                             confl[('%s_2' % k)] = f2_df[k].values
#
#                         confln =align_conflict(confl)
#
#                         if confln:
#
#                             confln['flight_id_1'] = f1['flight_id']
#                             confln['flight_id_2'] = f2['flight_id']
#
#                             try:
#                                 sql_inj_lst.append(tuple(confln[kk]
#                                                          for kk
#                                                          in col_lst))
#                             except Exception as e1:
#                                 print(e1)
#
#     if sql_inj_lst:
#         save_to_pg(sql_inj_lst, "conflicts")
#
#     else:
#         print('No conflicts found')
#
#
# def get_expected_conflicts(ep):
#
#     alt_min = flight_level_min
#     sql_inj_lst = []
#
#     col_lst = ['flight_id_1', 'ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1',
#                'hdg_1', 'roc_1', 'flight_id_2', 'ts_2', 'lat_2', 'lon_2',
#                'alt_2', 'spd_2', 'hdg_2', 'roc_2']
#
#     col_lst_2 = ['lat', 'lon', 'ts', 'hdg', 'alt', 'roc', 'spd']
#
#     batch = get_hourly_flight_batch(ep)
#
#     id_lst = [b['flight_id'] for b in batch]
#     fset = list(itertools.combinations(id_lst, 2))
#
#     fset_len = len(fset)
#     print('Set length: %d' % fset_len)
#
#     cnt = 0
#
#     for fs in fset:
#         cnt = cnt + 1
#
#         f1 = [f for f in batch if f['flight_id'] == fs[0]][0]
#         f2 = [f for f in batch if f['flight_id'] == fs[1]][0]
#
#         f1crd, f2crd, f1td, f2td = resample_flight_set(f1, f2, alt_min)
#
#         if not all(i for i in [f1crd, f2crd, f1td, f2td]):
#             continue
#
#         if f1td > la_time and f2td > la_time:
#
#             fi1, fi2 = find_flight_intersect([v[:3] for v in f1crd],
#                                              [v[:3] for v in f2crd])
#
#             if not fi1 and not fi2:
#
#                 confl_flag = check_path_intersection((f1crd[0][0],
#                                                       f1crd[0][1]),
#                                                      (f2crd[0][0],
#                                                       f2crd[0][1]),
#                                                      f1crd[0][3],
#                                                      f2crd[0][3])
#
#                 if confl_flag:
#                     confl = {}
#
#                     for ki in range(0, 7):
#
#                         confl[('%s_1' % col_lst_2[ki])] = \
#                             [v[ki] for v in f1crd]
#
#                         confl[('%s_2' % col_lst_2[ki])] = \
#                             [v[ki] for v in f2crd]
#
#                     confl_aligned = align_exp_conflict(confl)
#
#                     if confl_aligned:
#                         confl_aligned['flight_id_1'] = fs[0]
#                         confl_aligned['flight_id_2'] = fs[1]
#
#                         try:
#                             sql_inj_lst.append(tuple(confl_aligned[kk]
#                                                      for kk
#                                                      in col_lst))
#                         except Exception as e1:
#                             print(e1)
#
#     if sql_inj_lst:
#         save_to_pg(sql_inj_lst, 'exp_conflicts')
#
#     else:
#         print('No conflicts found')
#
#
# def get_overlapping_flights(ep):
#
#     alt_min = flight_level_min
#     sql_inj_lst = []
#     col_lst = ['flight_id_1', 'ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1',
#                'hdg_1', 'roc_1', 'flight_id_2', 'ts_2', 'lat_2', 'lon_2',
#                'alt_2', 'spd_2', 'hdg_2', 'roc_2']
#
#     col_lst_2 = ['lat', 'lon', 'ts', 'hdg', 'alt', 'roc', 'spd']
#
#     batch = get_hourly_flight_batch(ep)
#
#     id_lst = [b['flight_id'] for b in batch]
#     fset = list(itertools.combinations(id_lst, 2))
#
#     fset_len = len(fset)
#     print('Set length: %d' % fset_len)
#
#     cnt = 0
#
#     for fs in fset:
#         cnt = cnt + 1
#
#         f1 = [f for f in batch if f['flight_id'] == fs[0]][0]
#         f2 = [f for f in batch if f['flight_id'] == fs[1]][0]
#
#         f1crd, f2crd, f1td, f2td = resample_flight_set(f1, f2, alt_min)
#
#         if not all(i for i in [f1crd, f2crd, f1td, f2td]):
#             continue
#
#         if f1td > la_time and f2td > la_time:
#
#             start_dst = calc_coord_dst((f1crd[0][0], f1crd[0][1]),
#                                        (f2crd[0][0], f2crd[0][1]))
#
#             if start_dst < (f1crd[0][6] * knts_ms_ratio * max_t_startpoints):
#
#                 fl = {}
#
#                 for ki in range(0, 7):
#
#                     fl[('%s_1' % col_lst_2[ki])] = [v[ki] for v in f1crd]
#                     fl[('%s_2' % col_lst_2[ki])] = [v[ki] for v in f2crd]
#
#                 fl_aligned = align_exp_conflict(fl)
#
#                 if fl_aligned:
#                     fl_aligned['flight_id_1'] = fs[0]
#                     fl_aligned['flight_id_2'] = fs[1]
#
#                     try:
#                         sql_inj_lst.append(tuple(fl_aligned[kk]
#                                                  for kk in col_lst))
#
#                     except Exception as e1:
#                         print(e1)
#
#     if sql_inj_lst:
#         save_to_pg(sql_inj_lst, 'overlapping_flights')
#
#     else:
#         print('No flights found')


if __name__=="__main__":
    classify_flight_pairs(1526011200)
