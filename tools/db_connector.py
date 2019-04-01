import psycopg2 as psql
from conf.config import db_table_columns, flight_df_minlen


def get_pg_conn():

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' \
                            host='localhost' password='postgres'")
        return conn

    except Exception as e:
        print("Unable to connect to the database.")
        print(e)
        return None


def save_to_pg_simple(objs, table_name):

    conn = get_pg_conn()
    column_lst = db_table_columns[table_name]

    sql_inj_list = [tuple([d[k] for k in column_lst])
                    for d in objs]

    records_list_template = ','.join(['%s'] * len(objs))

    insert_query = 'insert into {} {} values {}'.format(
                    table_name, tuple(column_lst),
                    records_list_template).replace("'", "")

    try:
        cur = conn.cursor()
        cur.execute(insert_query, sql_inj_list)
        cur.close()
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print('Saving to DB failed, error: ')
        print(e)
        conn.close()
        return False


def save_to_pg(objs, table_name):

    conn = get_pg_conn()
    column_lst = db_table_columns[table_name]

    max_curs_len = 1000
    cnt = 0
    sub_list = []

    if len(objs) > 0:

        for o in objs:

            if cnt < max_curs_len:
                sub_list.append(o)
                cnt = cnt + 1
            else:
                sub_list.append(o)
                sql_inj_list = [tuple([d[k] for k in column_lst])
                                for d in sub_list]

                records_list_template = ','.join(['%s'] * len(sub_list))

                insert_query = 'insert into {} {} values {}'.format(
                                table_name, tuple(column_lst),
                                records_list_template)

                try:
                    cur = conn.cursor()
                    cur.execute(insert_query, sql_inj_list)
                    conn.commit()
                    cur.close()
                    sub_list = []

                except Exception as e:
                    print('Saving to DB failed, error: ')
                    print(e)
                    continue

                cnt = 0

        sql_inj_list = [tuple([d[k] for k in column_lst])
                        for d in sub_list]

        records_list_template = ','.join(['%s'] * len(sub_list))

        insert_query = 'insert into {} {} values {}'.format(
                        table_name, tuple(column_lst),
                        records_list_template).replace("'", "")

        try:
            cur = conn.cursor()
            cur.execute(insert_query, sql_inj_list)
            cur.close()
            conn.commit()
            return True

        except Exception as e:
            print('Saving to DB failed, error: ')
            print(e)
            return False


def save_adsb_flights_to_db(df, coll='adsb_flights'):
    dct_list = []
    for name, g in df.groupby('flight_id'):
        fl_data = g[['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc']]
        fl_data = fl_data.reset_index()

        if len(fl_data) < flight_df_minlen:
            # print("Flight not saved to db: DF has too few data points")
            continue

        try:
            dct = {
                "flight_length": len(g),
                "icao": g['icao'].dropna().unique()[0],
                "flight_id": g['flight_id'].dropna().unique()[0],
                "alt_min": fl_data['alt'].min(),
                "alt_max": fl_data['alt'].max(),
                "start_lat": fl_data['lat'].dropna().iloc[0],
                "start_lon": fl_data['lon'].dropna().iloc[0],
                "end_lat": fl_data['lat'].dropna().iloc[-1],
                "end_lon": fl_data['lon'].dropna().iloc[-1],
                "start_ep": fl_data['ts'].min(),
                "end_ep": fl_data['ts'].max(),
                "callsign": g['callsign'].dropna().unique()[0].strip('_'),
                "flight_number": int(g['flight'].dropna().unique()[0]),
                "ts": fl_data['ts'].tolist(),
                "lat": fl_data['lat'].tolist(),
                "lon": fl_data['lon'].tolist(),
                "alt": fl_data['alt'].tolist(),
                "spd": fl_data['spd'].tolist(),
                "hdg": fl_data['hdg'].tolist(),
                "roc": fl_data['roc'].tolist()
            }
            dct_list.append(dct)

        except Exception as e:
            print("Creating dict failed")
            continue

    try:
        if dct_list:
            save_to_pg(dct_list, coll)

    except Exception as e:
        print("Saving to DB failed")
        print(e)

    return True


def save_ddr2_flights_to_db(df_in, coll='ddr2_flights'):
    df = df_in
    try:
        dct = {
            "flight_length": len(df),
            "data_type": df['data_type'].dropna().unique()[0],
            "flight_id": int(df['flight_id'].dropna().unique()[0]),
            "org": df['org'].dropna().unique()[0],
            "dst": df['dst'].dropna().unique()[0],
            "start_ep": df['ep_seg_b'].min(),
            "end_ep": df['ep_seg_b'].max(),
            "ac_type": df['ac_type'].dropna().unique()[0],
            "callsign": df['callsgn'].dropna().unique()[0],
            "seg_id": df['seg_id'].tolist(),
            "ep_seg_b": df['ep_seg_b'].tolist(),
            "ep_seg_e": df['ep_seg_e'].tolist(),
            "fl_seg_b": df['fl_seg_b'].tolist(),
            "fl_seg_e": df['fl_seg_e'].tolist(),
            "status": df['status'].tolist(),
            "lat_seg_b": df['lat_seg_b'].tolist(),
            "lon_seg_b": df['lon_seg_b'].tolist(),
            "lat_seg_e": df['lat_seg_e'].tolist(),
            "lon_seg_e": df['lon_seg_e'].tolist(),
            "seq": df['seq'].tolist(),
            "seg_len": df['seg_len'].tolist(),
            "seg_par": df['seg_par'].tolist(),

            "start_lat": df.loc[df['seq'] == df['seq'].min(),
                                'lat_seg_b'].iloc[0],

            "start_lon": df.loc[df['seq'] == df['seq'].min(),
                                'lon_seg_b'].iloc[0],

            "end_lat": df.loc[df['seq'] == df['seq'].max(),
                              'lat_seg_b'].iloc[0],

            "end_lon": df.loc[df['seq'] == df['seq'].max(),
                              'lon_seg_b'].iloc[0],
        }

        try:
            save_to_pg([dct], coll)
            print("Flight saved to DB")

        except Exception as e:
            print("Saving to DB failed")

    except Exception as e:
        print("Creation of flight dict failed")

    return True
