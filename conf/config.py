
# Computational variables
cpu_count = 1


# Flight_filters
segmentation_window = 240
la_time = 1200
avg_sec_msg = 10
flight_level_min = 25000
flight_df_minlen = 100
ts_tol = 5
adsb_gap_filter_window = 3
max_adsb_gap = 20
min_fl_dataframe_length = la_time/max_adsb_gap


# Conflict filtering
max_confl_time_delta = 60
t_overshoot_max = 60
ttc_est_time_window = 1200
max_t_startpoints = 2400


# Global variables
earth_radius = 6378.1  # in km
earth_radius_m = earth_radius * 1000
knts_ms_ratio = 0.51444
ipz_range = 5 * 1852

sil_lat = 51.990
sil_lon = 4.375


# Database related lists
db_table_columns = {
    'adsb_flights': ["flight_length", "icao", "flight_id", "alt_min",
                     "alt_max", "start_lat", "start_lon", "end_lat",
                     "end_lon", "start_ep", "end_ep", "callsign",
                     "flight_number", "ts", "lat", "lon", "alt", "spd",
                     "hdg", "roc"],

    'adsb_flights_v2': ["flight_length", "icao", "flight_id", "alt_min",
                     "alt_max", "start_lat", "start_lon", "end_lat",
                     "end_lon", "start_ep", "end_ep", "callsign",
                     "flight_number", "ts", "lat", "lon", "alt", "spd",
                     "hdg", "roc"],

    'ddr2_flights': ["flight_length", "data_type", "flight_id", "org", "dst",
                     "start_ep", "end_ep", "ac_type", "callsign", "seg_id",
                     "ep_seg_b", "ep_seg_e", "fl_seg_b", "fl_seg_e", "status",
                     "lat_seg_b", "lon_seg_b", "lat_seg_e", "lon_seg_e",
                     "seq", "seg_len", "seg_par", "start_lat", "start_lon",
                     "end_lat", "end_lon"],

    'allft_flights': ["flight_length", "data_type", "flight_id", "org", "dst",
                     "start_ep", "end_ep", "ac_type", "callsign", "seg_id",
                     "ep_seg_b", "ep_seg_e", "fl_seg_b", "fl_seg_e", "status",
                     "lat_seg_b", "lon_seg_b", "lat_seg_e", "lon_seg_e",
                     "seq", "seg_len", "seg_par", "start_lat", "start_lon",
                     "end_lat", "end_lon"],

    'ctfm_flights': ["flight_length", "data_type", "flight_id", "org", "dst",
                     "start_ep", "end_ep", "ac_type", "callsign", "seg_id",
                     "ep_seg_b", "ep_seg_e", "fl_seg_b", "fl_seg_e", "status",
                     "lat_seg_b", "lon_seg_b", "lat_seg_e", "lon_seg_e",
                     "seq", "seg_len", "seg_par", "start_lat", "start_lon",
                     "end_lat", "end_lon"],

    'projected_flights': ["flight_length", "icao", "flight_id", "alt_min",
                          "alt_max", "start_lat", "start_lon", "end_lat",
                          "end_lon", "start_ep", "end_ep", "callsign",
                          "flight_number", "ts", "lat", "lon", "alt", "spd",
                          "hdg", "roc", "time_el", "proj_lat", "proj_lon",
                          "prev_lat", "prev_lon", "cte", "ate", "tte",
                          "time_proj"],

    'conflicts': ['td', 'altd', 'dstd', 'hdgd', 'flight_id_1', 'ts_1',
                  'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1',
                  'roc_1', 'flight_id_2', 'ts_2', 'lat_2', 'lon_2',
                  'alt_2', 'spd_2', 'hdg_2', 'roc_2'],

    'converging_flights': ['td', 'altd', 'dstd', 'hdgd', 'flight_id_1',
                      'ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1',
                      'hdg_1', 'roc_1', 'flight_id_2', 'ts_2', 'lat_2',
                      'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2'],

    'overlapping_flights': ['td', 'altd', 'dstd', 'hdgd', 'flight_id_1',
                            'ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1',
                            'hdg_1', 'roc_1', 'flight_id_2', 'ts_2', 'lat_2',
                            'lon_2', 'alt_2', 'spd_2', 'hdg_2', 'roc_2'],
}

input_data_columns = {
    'ddr2': ['seg_id', 'org', 'dst', 'ac_type', 't_seg_b', 't_seg_e',
             'fl_seg_b', 'fl_seg_e', 'status', 'callsgn',
             'dd_seg_b', 'dd_seg_e', 'lat_seg_b', 'lon_seg_b',
             'lat_seg_e', 'lon_seg_e', 'flight_id', 'seq', 'seg_len',
             'seg_par'],

    'adsb_old': ['ts', 'icao', 'tc', 'msg'],

    'adsb': ['ix', 'ts', 'tc', 'icao', 'msg', 'df']

}


# ------from flight_conflict_processor------ #
# conn_write = get_pg_conn()
# cur_inj = conn_write.cursor()
# records_list_template = ','.join(['%s'] * len(sql_inj_lst))
#
# insert_query = 'insert into overlapping_flights (flight_id_1, ts_1, \
#                 lat_1, lon_1, alt_1, spd_1, hdg_1, roc_1,\
#                 flight_id_2, ts_2, lat_2, lon_2, alt_2, spd_2, \
#                 hdg_2, roc_2) values {}'.format(records_list_template)
#
# cur_inj.execute(insert_query, sql_inj_lst)
# conn_write.commit()
# print("%d Flights inserted" % len(sql_inj_lst))
# cur_inj.close()
# conn_write.close()

