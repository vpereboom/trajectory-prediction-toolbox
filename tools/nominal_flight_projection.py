import pandas as pd
import psycopg2 as psql
from psycopg2.extras import RealDictCursor
import time

from tools.flight_projection import *

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

    la_time = 1400
    cnt = 0
    cnt_max = np.inf
    fl_len_min = la_time*0.6
    flight_level_min = 15000

    column_order_lst_proj = ["flight_length", "icao", "flight_id", "alt_min", "alt_max", "start_lat", "start_lon", "end_lat",
                        "end_lon", "start_ep", "end_ep", "callsign", "flight_number", "ts", "lat", "lon", "alt", "spd",
                        "hdg", "roc", "time_el", "proj_lat", "proj_lon", "prev_lat", "prev_lon", "cte", "ate", "tte",
                        "time_proj"]

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("I am unable to connect to the database.")
        print(e)

    max_inserts = 500
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

            if fl_d['time_el'].max() < fl_len_min:
                print("flight too short")
                continue

            try:
                steps = int(fl_d['time_el'].max() / la_time) + 1

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