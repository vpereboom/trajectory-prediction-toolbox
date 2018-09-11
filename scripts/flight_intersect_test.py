import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import json
from scipy import stats
import psycopg2 as psql
from psycopg2.extras import RealDictCursor

import seaborn as sns
sns.set(color_codes=True)


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


if __name__ == "__main__":

    try:
        conn = psql.connect("dbname='thesisdata' user='postgres' host='localhost' password='postgres'")
    except Exception as e:
        print("Unable to connect to the database.")
        print(e)

    max_inserts = 100
    fetch_batch_size = max_inserts
    cnt = 0

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT ts, lat, lon, alt, start_ep FROM public.adsb_flights WHERE flight_length > 1000 LIMIT 50;")

    fetch_batch_size = 500
    ts_offset = 3600
    max_dst = 5 * 1852
    max_ts = 30
    alt_min = 15000

    f_list = []
    while True:

        batch = cur_read.fetchall()

        if not batch:
            break

        for f1 in batch:

            # f1 = {k: v for k, v in f1.items() if }
            f1crd = [(lt, ln, ts) for lt, ln, alt, ts in list(zip(f1['lat'], f1['lon'], f1['alt'], f1['ts'])) if alt > alt_min]

            fl1_start_ep = f1['start_ep']

            cur_read_2 = conn.cursor(cursor_factory=RealDictCursor)
            cur_read_2.execute("SELECT ts, lat, lon, alt, start_ep FROM public.adsb_flights WHERE start_ep BETWEEN %s AND %s LIMIT 500;",
                               (fl1_start_ep - ts_offset, fl1_start_ep + ts_offset))

            while True:

                batch_2 = cur_read_2.fetchall()

                if not batch_2:
                    break

                for f2 in batch_2:
                    # f1crd = list(zip(f1['lat'], f1['lon']))
                    # f2crd = list(zip(f2['lat'], f2['lon']))
                    f2crd = [(lt, ln, ts) for lt, ln, alt, ts in list(zip(f2['lat'], f2['lon'], f2['alt'], f2['ts'])) if
                             alt > alt_min]

                    if len(f2crd) > 20 and len(f1crd) > 20:

                        t1 = time.time()

                        fi1, fi2 = find_flight_intersect(f1crd, f2crd)

                        if fi1:
                            # print(len(fi1))
                            d, c1, c2, i1, i2 = closest_distance(list(fi1), list(fi2))

                            tdiff = abs(fi1[i1][2] - fi2[i2][2])
                            if d < max_dst and tdiff < max_ts:
                                print((d, c1, c2))
                                print(tdiff)
                                print("Calc took: %f seconds" % (time.time()-t1))

    cur_read.close()
    cur_read_2.close()