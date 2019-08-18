import math
import numpy as np
import pandas as pd
from operator import itemgetter
from conf.config import earth_radius, earth_radius_m, knts_ms_ratio


def find_coord_dst_hdg(coord1, hdg, dst):

    # https://stackoverflow.com/questions/7222382/
    # get-lat-long-given-current-point-distance-and-bearing

    hdg = math.radians(hdg)         # Bearing is converted to radians.
    d = dst / 1000                  # Distance in km

    lat1 = math.radians(coord1[0])  # Current lat point converted to radians
    lon1 = math.radians(coord1[1])  # Current long point converted to radians

    lat2 = math.asin(math.sin(lat1) * math.cos(d / earth_radius) +
        math.cos(lat1) * math.sin(d / earth_radius) * math.cos(hdg))

    y = math.sin(hdg) * math.sin(d / earth_radius) * math.cos(lat1)
    x = math.cos(d / earth_radius) - math.sin(lat1) * math.sin(lat2)

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

                dst_start = (row[r['ts']] - ts_start) * \
                            (spd_start * knts_ms_ratio)

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

    coord_start = (fl_df['lat'].iloc[hdg_start_nr],
                   fl_df['lon'].iloc[hdg_start_nr])

    hdg_avg_start = fl_df['hdg'].iloc[list(range(0, hdg_start_nr))].mean()
    spd_avg_start = fl_df['spd'].iloc[list(range(0, hdg_start_nr))].mean()

    ts_start = fl_df['ts'].iloc[hdg_start_nr]
    fl_df['proj_lat'] = np.nan
    fl_df['proj_lon'] = np.nan

    for i in range(hdg_start_nr, len(fl_df)):
        if (fl_df['ts'].iloc[i] - ts_start) < look_ahead_t:

            dst_start = (fl_df['ts'].iloc[i] - ts_start) * \
                        (spd_avg_start * knts_ms_ratio)

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

    lat1 = c1[0]
    lon1 = c1[1]
    lat2 = c2[0]
    lon2 = c2[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l
                                in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = earth_radius_m * c
    return d


def calc_coord_dst_simple(c1, c2):

    lon1 = c1[0]
    lat1 = c1[1]
    lon2 = c2[0]
    lat2 = c2[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l
                                in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    x = dlon * math.cos(dlat / 2)
    y = dlat
    d = math.sqrt(x * x + y * y) * earth_radius_m

    return d


def calc_bearing(c0, c1):
    if not all(isinstance(i, tuple) for i in [c0, c1]):
        return None

    lat1 = c0[0]
    lon1 = c0[1]
    lat2 = c1[0]
    lon2 = c1[1]

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l
                                in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1

    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
        math.cos(lat2) * math.cos(dlon)

    y = math.sin(dlon) * math.cos(lat2)
    bearing = math.atan2(y, x)

    return math.degrees(bearing)


def calc_compass_bearing(c0, c1):
    bearing = calc_bearing(c0, c1)

    return (bearing + 360) % 360


def get_triangle_corner(d0, d1, d2):
    bb = (d0 ** 2 + d1 ** 2 - d2 ** 2) / (2 * d0 * d1)
    return math.acos(bb)


def convert_knts_ms(spd):
    return spd*knts_ms_ratio


def heading_diff(h1, h2):

    r = (h2 - h1) % 360.0

    if r >= 180.0:
        r -= 360.0

    return r


def calc_track_errors(wp_last, wp_curr, wp_proj):
    if not all(isinstance(i, tuple) for i in [wp_last, wp_curr, wp_proj]):
        return None, None, None

    consec_range_lim = 100000
    if abs(calc_coord_dst_simple(wp_last, wp_curr)) > consec_range_lim:
        print('Anomalous value')
        return None, None, None

    dst_proj_curr = calc_coord_dst_simple(wp_curr, wp_proj)

    hdg_curr_proj = calc_compass_bearing(wp_curr, wp_proj)
    hdg_curr_last = calc_compass_bearing(wp_curr, wp_last)

    alpha = heading_diff(hdg_curr_proj, hdg_curr_last)
    tte = dst_proj_curr

    if abs(alpha) < 90:
        cte = math.sin(math.radians(alpha)) * tte
        ate = -math.sqrt(tte ** 2 - cte ** 2)

    else:
        if alpha < 0:
            cte = math.sin(math.radians(-180 - alpha)) * tte
            ate = math.sqrt(tte ** 2 - cte ** 2)
        else:
            cte = math.sin(math.radians(180 - alpha)) * tte
            ate = math.sqrt(tte ** 2 - cte ** 2)

    return cte, ate, tte


def check_ddr2_wps(lat_1, lon_1, lat_2, lon_2, lat_ac, lon_ac):
    hdg_1 = calc_compass_bearing((lat_1, lon_1), (lat_ac, lon_ac))
    hdg_2 = calc_compass_bearing((lat_2, lon_2), (lat_ac, lon_ac))

    deg_f1_f2 = calc_compass_bearing((lat_1, lon_1), (lat_2, lon_2))

    alpha = (360 + (hdg_1 - deg_f1_f2)) % 360
    beta = (360 + (hdg_2 - deg_f1_f2)) % 360

    if alpha > beta > 180:
        alpha_mod = 360 - alpha
        beta_mod = beta % 180

    elif alpha < beta < 180:
        alpha_mod = alpha
        beta_mod = 180 - beta

    else:
        return None

    if alpha_mod < 90 and beta_mod < 90:
        dst = calc_coord_dst((lat_1, lon_1), (lat_ac, lon_ac))
        return math.sin(math.radians(alpha_mod)) * dst
    else:
        return None


def evaluate_triangle(wp_1, wp_2, wp_ac):

    [d_12, d_ac1, d_ac2] = [calc_coord_dst(wp_1, wp_2),
                            calc_coord_dst(wp_1, wp_ac),
                            calc_coord_dst(wp_2, wp_ac)]

    alpha_1 = get_triangle_corner(d_12, d_ac1, d_ac2)
    alpha_2 = get_triangle_corner(d_12, d_ac2, d_ac1)
    alpha_ac = get_triangle_corner(d_ac2, d_ac1, d_12)

    return alpha_1, alpha_2, alpha_ac


def evaluate_waypoints(wp_ac, last_wp, curr_wp, next_wp):

    [wp_11, wp_12, wp_21, wp_22] = [last_wp, curr_wp, curr_wp, next_wp]

    a_11, a_12, a_1ac = evaluate_triangle(wp_11, wp_12, wp_ac)
    a_21, a_22, a_2ac = evaluate_triangle(wp_21, wp_22, wp_ac)

    if a_12 > a_21:
        return True
    else:
        return False


def find_closest_waypoint_index(wp, wp_list):
    dst = [calc_coord_dst(wp, wpi) for wpi in wp_list]
    ix = dst.index(min(dst))

    return ix


def get_triangle_height(d0, d1, d2):
    x = (d2 ** 2 - d1 ** 2 + d0 ** 2) / (2 * d0)
    h = math.sqrt(d2 ** 2 - x ** 2)

    return h


def find_waypoint_index(wp, wp_list):
    wp_ac = wp
    hi_lst = []

    for i, w in enumerate(wp_list[:-1]):
        wp_0 = w
        wp_1 = wp_list[i + 1]

        a_0, a_1, a_ac = evaluate_triangle(wp_0, wp_1, wp_ac)

        if (abs(a_0) < 90) & (abs(a_1) < 90):
            d0 = calc_coord_dst(wp_0, wp_1)
            d1 = calc_coord_dst(wp_1, wp_ac)
            d2 = calc_coord_dst(wp_0, wp_ac)

            hi = get_triangle_height(d0, d1, d2)
            hi_lst.append((i, hi))

    if hi_lst:
        iwp = min(hi_lst, key=lambda t: t[1])[0]
    else:
        iwp = find_closest_waypoint_index(wp, wp_list)

    return iwp


def find_waypoint_index_v2(wp, wp_list):
    wp_ac = wp
    hi_lst = []

    for i, w in enumerate(wp_list[:-1]):
        wp_0 = w
        wp_1 = wp_list[i + 1]

        dst_i = check_ddr2_wps(wp_0[0], wp_0[1], wp_1[0], wp_1[1], wp_ac[0],
                               wp_ac[1])

        if dst_i:
            hi_lst.append((i, dst_i))

    if hi_lst:
        iwp = min(hi_lst, key=lambda t: t[1])[0]
    else:
        iwp = find_closest_waypoint_index(wp, wp_list)

    return iwp


def find_wp_segment(wp_ac, seg_list):
    hi_lst = []

    for i, seg in enumerate(seg_list):

        dst_i = check_ddr2_wps(seg[0], seg[1], seg[2], seg[3], wp_ac[0],
                               wp_ac[1])

        if dst_i:
            hi_lst.append((i, dst_i, seg))

    hi_lst = [i for i in hi_lst if i]

    if hi_lst:
        curr_seg = hi_lst[-1]  # min(hi_lst, key=lambda t: t[1])[0]
    else:
        return None

    return curr_seg


def add_waypoints_seq(flight_df, route_wps):

    # Reset index to start at 0
    # Ensure all lat and lon values exist in the used rows

    flight_df = flight_df[(flight_df['lat'].notnull() &
                           flight_df['lon'].notnull())]

    flight_df = flight_df.reset_index(drop=True)
    route_wps = route_wps.reset_index(drop=True)

    last_wp_list = [0] * len(flight_df)
    curr_wp_list = [0] * len(flight_df)
    next_wp_list = [0] * len(flight_df)
    wp_ac_list = [0] * len(flight_df)

    for i, r in flight_df.iterrows():

        wp_ac = (r['lat'], r['lon'])

        if i == 0:
            #             last_wp = (r['lat'], r['lon'])
            # last_wp_ix = find_closest_waypoint_index(wp_ac, route_wps)
            last_wp_ix = find_waypoint_index_v2(wp_ac, route_wps)
            curr_wp_i = last_wp_ix + 1

        last_wp = route_wps[curr_wp_i - 1]
        curr_wp = route_wps[curr_wp_i]
        next_wp = route_wps[curr_wp_i + 1]

        last_wp_list[i] = last_wp
        curr_wp_list[i] = curr_wp
        next_wp_list[i] = next_wp
        wp_ac_list[i] = wp_ac

        switch = evaluate_waypoints(wp_ac, last_wp, curr_wp, next_wp)

        if switch:
            if curr_wp_i < (len(route_wps) - 2):
                # last_wp = route_wps[curr_wp_i]
                curr_wp_i = curr_wp_i + 1
            else:
                flight_df['last_wp'] = last_wp_list
                flight_df['curr_wp'] = curr_wp_list
                flight_df['next_wp'] = next_wp_list
                flight_df['wp_ac'] = wp_ac_list

                return flight_df

    flight_df['last_wp'] = last_wp_list
    flight_df['curr_wp'] = curr_wp_list
    flight_df['next_wp'] = next_wp_list
    flight_df['wp_ac'] = wp_ac_list

    return flight_df


def find_wp_segments(flight_df, route_wps):

    # Reset index to start at 0
    # Ensure all lat and lon values exist in the used rows

    flight_df = flight_df.reset_index(drop=True)
    route_wps = route_wps.reset_index(drop=True)

    seg_list = [0] * len(flight_df)

    for i, r in flight_df.iterrows():

        wp_ac = (r['lat'], r['lon'])
        seg_list[i] = find_wp_segment(wp_ac, route_wps)

    flight_df['wp_seg'] = seg_list

    return flight_df


def add_waypoints_free(flight_df, route_wps):

    # Reset index to start at 0
    # Ensure all lat and lon values exist in the used rows

    flight_df = flight_df[(flight_df['lat'].notnull() &
                           flight_df['lon'].notnull())]

    flight_df = flight_df.reset_index(drop=True)
    route_wps = route_wps.reset_index(drop=True)

    last_wp_list = [0] * len(flight_df)
    curr_wp_list = [0] * len(flight_df)
    next_wp_list = [0] * len(flight_df)
    wp_ac_list = [0] * len(flight_df)

    for i, r in flight_df.iterrows():

        wp_ac = (r['lat'], r['lon'])

        curr_wp_i = find_waypoint_index(wp_ac, route_wps)

        if curr_wp_i == 0:
            curr_wp_i = 1

        if curr_wp_i < (len(route_wps) - 2):

            last_wp = route_wps[curr_wp_i - 1]
            curr_wp = route_wps[curr_wp_i]
            next_wp = route_wps[curr_wp_i + 1]

            last_wp_list[i] = last_wp
            curr_wp_list[i] = curr_wp
            next_wp_list[i] = next_wp
            wp_ac_list[i] = wp_ac

        else:
            flight_df['last_wp'] = last_wp_list
            flight_df['curr_wp'] = curr_wp_list
            flight_df['next_wp'] = next_wp_list
            flight_df['wp_ac'] = wp_ac_list

            return flight_df

    flight_df['last_wp'] = last_wp_list
    flight_df['curr_wp'] = curr_wp_list
    flight_df['next_wp'] = next_wp_list
    flight_df['wp_ac'] = wp_ac_list

    return flight_df


def add_waypoints(b):
    dc = {k: b[k] for k in ['ep_seg_b', 'lat_seg_b', 'lon_seg_b',
                            'lat_seg_e', 'lon_seg_e']}
    ac = {k: b[k] for k in ['ts', 'lat', 'lon', 'spd', 'hdg', 'alt']}

    dc = pd.DataFrame.from_dict(dc)
    ac = pd.DataFrame.from_dict(ac)

    dc['wps'] = list(zip(dc.lat_seg_b, dc.lon_seg_b, dc.lat_seg_e,
                         dc.lon_seg_e))
    try:
        df = add_waypoints_seq(ac, dc['wps'])
    except Exception as e:
        print(e)
        return None

    df[['curr_lat', 'curr_lon']] = df['curr_wp'].apply(pd.Series)
    df[['next_lat', 'next_lon']] = df['next_wp'].apply(pd.Series)
    df[['last_lat', 'last_lon']] = df['last_wp'].apply(pd.Series)

    return df


def add_waypoint_segments(b):
    dc = {k: b[k] for k in ['ep_seg_b', 'lat_seg_b', 'lon_seg_b',
                            'lat_seg_e', 'lon_seg_e']}
    ac = {k: b[k] for k in ['ts', 'lat', 'lon', 'spd', 'hdg', 'alt']}

    dc = pd.DataFrame.from_dict(dc)
    ac = pd.DataFrame.from_dict(ac)

    dc['wps'] = list(zip(dc.lat_seg_b, dc.lon_seg_b, dc.lat_seg_e,
                         dc.lon_seg_e))
    try:
        df = find_wp_segments(ac, dc['wps'])
    except Exception as e:
        print(e)
        return None

    return df


def calc_coord_dst_pp(lon1, lat1, lon2, lat2):

    [lon1, lat1, lon2, lat2] = [math.radians(l) for l
                                in [lon1, lat1, lon2, lat2]]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = earth_radius_m * c
    return d


def closest_distance(f1, f2):
    """Flights should be zipped list like zip(lat,lon)"""

    x = [[(calc_coord_dst(c1, c2), c1, c2) for c1 in f1] for c2 in f2]
    xflat = [item for sublist in x for item in sublist]
    tupmin = min(xflat, key=itemgetter(2))

    dmin = tupmin[0]
    c1min = tupmin[1][:2]
    c2min = tupmin[2][:2]

    return dmin, c1min, c2min


if __name__=="__main__":

    import psycopg2 as psql
    from psycopg2.extras import RealDictCursor
    from tools.conflict_handling import crop_ddr2_flight_seg

    try:
        conn = psql.connect(
            "dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("Unable to connect to the database.")
        print(e)

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT * FROM public.ddr2_conflicts limit 20;")
    batch = cur_read.fetchall()

    res_batch = []

    for bi in batch:

        b = {}

        if bi:
            fl_keys = ['ts', 'lat', 'lon', 'hdg', 'alt', 'spd',
                       'roc', 'ep_seg_b', 'lat_seg_b', 'lon_seg_b',
                       'lat_seg_e', 'lon_seg_e',
                       'fl_seg_b', 'fl_seg_e', 'seq']

            try:
                fl1 = {}
                for k in fl_keys:
                    fl1[k] = bi["%s%s" % (k, '_1')]

                try:
                    fl1 = crop_ddr2_flight_seg(fl1)
                except Exception as e:
                    print('Cropping DDR2 flight failed, error: ')
                    print(e)
                    continue

                if fl1 is None:
                    continue

                try:
                    fl1 = add_waypoint_segments(fl1)
                    fl1 = fl1[fl1.wp_seg.notnull()]
                except Exception as e:
                    print('Adding waypoints to flight failed, error: ')
                    print(e)
                    continue

                if fl1 is None:
                    continue

                if len(fl1) == 0:
                    continue

                fl1['hdg_int'] = fl1.apply(lambda x: calc_compass_bearing(
                                    (x['wp_seg'][2][0], x['wp_seg'][2][1]),
                                    (x['wp_seg'][2][2], x['wp_seg'][2][3])),
                                           axis=1)

            except Exception as e:
                print('Preprocessing data failed, error:')
                print(e)
                continue
