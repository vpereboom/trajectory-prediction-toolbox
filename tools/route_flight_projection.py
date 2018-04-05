import pandas as pd
import math
import numpy as np
import datetime
import time

from pymongo import MongoClient
cl = MongoClient()
db = cl['flights']
col = db['ddr2_flights']
col_combi = db['combi_ddr2_adsb']


def calc_coord_dst(c1, c2):
    R = 6378.1 * 1000  # Radius of the Earth in m

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


def get_triangle_corner(d0, d1, d2):
    bb = (d0 ** 2 + d1 ** 2 - d2 ** 2) / (2 * d0 * d1)
    return math.degrees(math.acos(bb))


def evaluate_triangle(wp_1, wp_2, wp_ac):

    [d_12, d_ac1, d_ac2] = [calc_coord_dst(wp_1, wp_2), calc_coord_dst(wp_1, wp_ac), calc_coord_dst(wp_2, wp_ac)]

    alpha_1 = get_triangle_corner(d_12, d_ac1, d_ac2)
    alpha_2 = get_triangle_corner(d_12, d_ac2, d_ac1)
    alpha_ac = get_triangle_corner(d_ac2, d_ac1, d_12)

    return alpha_1, alpha_2, alpha_ac


def evaluate_waypoints(wp_ac, last_wp, curr_wp, next_wp):

    [wp_11, wp_12, wp_21, wp_22] = [last_wp, curr_wp, curr_wp, next_wp]

    a_11, a_12, a_1ac = evaluate_triangle(wp_11, wp_12, wp_ac)
    a_21, a_22, a_2ac = evaluate_triangle(wp_21, wp_22, wp_ac)

    if all([abs(i) <= 90 for i in [a_11, a_12]]):
        if all([abs(i) <= 90 for i in [a_21, a_22]]):
            if abs(a_1ac) < abs(a_2ac):
                return False
            else:
                return True
        else:
            return False
    else:
        if all([abs(i) <= 90 for i in [a_21, a_22]]):
            return True
        else:
            return False


def find_closest_waypoint_index(wp, wp_list):

    dst = [calc_coord_dst(wp, wpi) for wpi in wp_list]
    ix = dst.index(min(dst))

    return ix


def add_waypoints(flight_df, route_wps):

    # Reset index to start at 0
    # Ensure all lat and lon values exist in the used rows
    flight_df = flight_df[(flight_df['lat'].notnull() & flight_df['lon'].notnull())]
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
            if curr_wp_i < (len(route_wps) -2):
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


if __name__ == '__main__':

    x = col_combi.find_one({})
    a = pd.DataFrame(x['adsb'])
    d = pd.DataFrame(x['ddr2'])

    dc = d[['ep_seg_b', 'lat_seg_b', 'lon_seg_b']]
    ac = a[['ts', 'lat', 'lon']]

    dc['wps'] = list(zip(dc.lat_seg_b, dc.lon_seg_b))

    df = add_waypoints(ac, dc['wps'])
    df