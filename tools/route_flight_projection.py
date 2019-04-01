import pandas as pd
import math
import numpy as np

from tools.nominal_flight_projection import calc_coord_dst
from tools.nominal_flight_projection import get_triangle_corner
from tools.nominal_flight_projection import calc_bearing
from tools.nominal_flight_projection import find_coord_dst_hdg


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
            last_wp_ix = find_closest_waypoint_index(wp_ac, route_wps)
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


def calc_track_errors(wp_0, wp_1, wp_ac):
    if not all(isinstance(i, tuple) for i in [wp_0, wp_1, wp_ac]):
        return np.nan, np.nan, np.nan

    dst_ac = calc_coord_dst(wp_0, wp_ac)
    a_0, a_1, a_ac = evaluate_triangle(wp_0, wp_1, wp_ac)

    hdg_wp0_wp1 = calc_bearing(wp_0, wp_1)
    hdg_wp0_wpac = calc_bearing(wp_0, wp_ac)

    wp_proj = find_coord_dst_hdg(wp_0, hdg_wp0_wp1, dst_ac)
    dst_wp_proj = calc_coord_dst(wp_0, wp_proj)
    dst_proj_ac = calc_coord_dst(wp_ac, wp_proj)

    hdg_wp0_wpac = calc_bearing(wp_0, wp_ac)
    alpha_2 = math.radians(hdg_wp0_wpac - hdg_wp0_wp1)

    try:
        cte = math.sin(alpha_2) * dst_ac
        ate = math.sqrt(dst_ac ** 2 - cte ** 2) - dst_ac
        tte = dst_proj_ac
    except Exception as e:
        print(e)
        cte = np.nan
        tte = np.nan
        ate = np.nan

    if dst_ac < dst_wp_proj:
        ate = -1 * ate

    return cte, ate, tte


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    proj_dct = {}
    plt.figure(figsize=(30, 30))
    bin_cte = []

    for x in col_combi.find({}):
        a = pd.DataFrame(x['adsb'])
        d = pd.DataFrame(x['ddr2'])

        dc = d[['ep_seg_b', 'lat_seg_b', 'lon_seg_b']]
        ac = a[['ts', 'lat', 'lon']]

        dc['wps'] = list(zip(dc.lat_seg_b, dc.lon_seg_b))
        try:
            df = add_waypoints_v2(ac, dc['wps'])
        except Exception as e:
            print(e)
            continue

        df[['curr_lat', 'curr_lon']] = df['curr_wp'].apply(pd.Series)
        df[['next_lat', 'next_lon']] = df['next_wp'].apply(pd.Series)
        df[['last_lat', 'last_lon']] = df['last_wp'].apply(pd.Series)

        cte_arr = []
        ate_arr = []
        tte_arr = []

        fl_dd = pd.DataFrame()

        for ii, r in df.iterrows():

            cte, ate, tte = calc_track_errors(r['last_wp'], r['curr_wp'],
                                              r['wp_ac'])

            cte_arr.append(cte)
            ate_arr.append(ate)
            tte_arr.append(tte)

        fl_dd['cte'] = cte_arr
        fl_dd['ate'] = ate_arr
        fl_dd['tte'] = tte_arr
        fl_dd['time_proj'] = df['ts'] - df['ts'].min()

        bin_cte.extend(cte_arr)

        plt.scatter(fl_dd['time_proj'], fl_dd['cte'], s=2, c='b')

    plt.show()
