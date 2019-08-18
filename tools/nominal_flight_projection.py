import pandas as pd
import numpy as np
import multiprocessing
from psycopg2.extras import RealDictCursor
import time

from tools.flight_projection import calc_track_errors, nominal_proj
from tools.db_connector import get_pg_conn
from conf.config import la_time, flight_level_min, avg_sec_msg

# TODO Implement metric conventions


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
            if not cte:
                return None
            else:
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

    dct = proj_df.to_dict(orient="list")

    for k in ["flight_length", "icao", "flight_id", "alt_min", "alt_max",
              "start_lat", "start_lon", "end_lat", "end_lon", "start_ep",
              "end_ep", "callsign", "flight_number"]:

        dct[k] = dct[k][0]

    return dct


def flush_to_db(insert_lst, conn):

    try:

        records_list_template = ','.join(['%s'] * len(insert_lst))

        insert_query = 'insert into projected_flights (flight_length, icao, \
                        flight_id, alt_min, alt_max, start_lat, start_lon, \
                        end_lat, end_lon, start_ep, end_ep, callsign, \
                        flight_number, ts, lat, lon, alt, spd, hdg, roc, \
                        time_el, proj_lat, proj_lon, prev_lat, prev_lon, \
                        cte, ate, tte, time_proj) \
                        values {}'.format(records_list_template)

        cur_insert = conn.cursor()

        sql_inj_list = [tuple([d[k] for k in column_order_lst_proj])
                        for d in insert_lst]

        cur_insert.execute(insert_query, sql_inj_list)
        conn.commit()
        cur_insert.close()

        return True

    except Exception as e:
        print("Flush to DB failed:")
        print(e)

        return False


def process_flights(sub_batch):

    db_conn = get_pg_conn()

    insert_lst = []

    for fl in sub_batch:
        fl_d = pd.DataFrame.from_dict(fl)
        fl_d = fl_d[fl_d['alt'] > flight_level_min]

        if (fl_d['ts'].max() - fl_d['ts'].min()) < la_time:
            # print("Flight too short above selected FL")
            continue

        else:

            fl_d['time_el'] = fl_d['ts'] - fl_d['ts'].min()

            # Clean up the dataframe by patching NAN values
            fl_d['hdg'] = fl_d['hdg'].fillna(method='ffill')
            fl_d['spd'] = fl_d['spd'].fillna(method='ffill')

            try:
                steps = int(fl_d['time_el'].max() / la_time) + 1

                for i in range(steps):
                    dct = create_projection_dict(fl_d[(fl_d['time_el'] >
                                                       i * la_time) &
                                                      (fl_d['time_el'] <
                                                       (i + 1) * la_time)],
                                                 la_time)
                    if dct:
                        insert_lst.append(dct)

            except Exception as e:
                print(e)
                continue

    # Flush to DB
    flush_stat = flush_to_db(insert_lst, db_conn)
    if flush_stat:
        print("%d Flights inserted" % len(insert_lst))

    return True


if __name__ == "__main__":

    cnt = 0
    cnt_max = np.inf
    limit = 5000
    offset = 0
    limit_d = limit
    cnt = 0
    flight_limit = 20000

    column_order_lst_proj = ["flight_length", "icao", "flight_id", "alt_min",
                             "alt_max", "start_lat", "start_lon", "end_lat",
                             "end_lon", "start_ep", "end_ep", "callsign",
                             "flight_number", "ts", "lat", "lon", "alt",
                             "spd", "hdg", "roc", "time_el", "proj_lat",
                             "proj_lon", "prev_lat", "prev_lon", "cte",
                             "ate", "tte", "time_proj"]

    conn = get_pg_conn()

    while True:

        t0 = time.time()
        if offset > flight_limit:
            break

        cur_read = conn.cursor(cursor_factory=RealDictCursor)

        cur_read.execute("SELECT * FROM public.adsb_flights \
                         WHERE (end_ep - start_ep) > %d \
                         AND ((end_ep - start_ep) / flight_length) < %d \
                         ORDER BY start_ep OFFSET %d LIMIT %d;"
                         % (la_time, avg_sec_msg, offset, limit))

        offset = offset + limit_d

        batch = cur_read.fetchall()

        if not batch:
            print('Result query empty')
            break

        pool_cpu_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(pool_cpu_size)
        batch_part_len = int(len(batch)/pool_cpu_size)

        batch_part_list = [batch[i*batch_part_len:(i+1)*batch_part_len]
                           for i in range(0, pool_cpu_size + 1)
                           if len(batch[i * batch_part_len:(i+1) *
                                        batch_part_len]) > 0]

        res = pool.map(process_flights, batch_part_list)
        pool.close()
        pool.join()

        cur_read.close()

        print('%d Flights inserted in %d seconds' % (len(batch),
                                                     int(time.time()-t0)))

    print("Done")
