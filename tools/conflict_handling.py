import numpy as np
import math
import pickle
from copy import copy

from tools.flight_projection import calc_coord_dst, calc_compass_bearing, \
                                    add_waypoints, find_waypoint_index_v2, \
                                    find_wp_segment, add_waypoint_segments, \
                                    find_coord_dst_hdg, heading_diff

from conf.config import ipz_range, knts_ms_ratio, t_overshoot_max, \
                        ttc_est_time_window, la_time


def load_obj(name):
    return pickle.load(open(name, "rb"))


def gamma_angle(lat_1, lon_1, lat_2, lon_2, hdg_1, hdg_2):

    deg_f1_f2 = calc_compass_bearing((lat_1, lon_1), (lat_2, lon_2))

    alpha = (360 + (hdg_1 - deg_f1_f2)) % 360
    beta = (360 + (hdg_2 - deg_f1_f2)) % 360

    if alpha > beta > 180:
        gamma = alpha - beta
        alpha_mod = 360 - alpha
        beta_mod = beta % 180

        return gamma, alpha_mod, beta_mod

    elif alpha < beta < 180:
        gamma = beta - alpha
        alpha_mod = alpha
        beta_mod = 180 - beta

        return gamma, alpha_mod, beta_mod

    else:
        return np.nan, np.nan, np.nan


def get_ac_distance_time(s, dx1, dx2, dy1, dy2, t):

    f1d = np.array([0, 0]) + np.array([dx1 * t, dy1 * t])
    f2d = np.array([s, 0]) + np.array([dx2 * t, dy2 * t])
    fdiff = f1d - f2d
    arg = fdiff[0] ** 2 + fdiff[1] ** 2
    d = np.sqrt(arg)

    return d


def get_dx_dy(a, b, spd_1, spd_2):

    b = 180-b
    dy_1 = math.sin(math.radians(a)) * spd_1 * knts_ms_ratio
    dy_2 = math.sin(math.radians(b)) * spd_2 * knts_ms_ratio
    dx_1 = math.cos(math.radians(a)) * spd_1 * knts_ms_ratio
    dx_2 = math.cos(math.radians(b)) * spd_2 * knts_ms_ratio

    return dx_1, dx_2, dy_1, dy_2


def get_delta_dst(lat_1, lon_1, lat_2, lon_2, hdg_1, hdg_2, spd_1, spd_2):

    _gamma, _alpha, _beta = gamma_angle(lat_1, lon_1, lat_2, lon_2, hdg_1,
                                        hdg_2)

    if np.isnan(_gamma):
        return np.nan
    else:
        dx_1, dx_2, dy_1, dy_2 = get_dx_dy(_alpha, _beta, spd_1, spd_2)
        s_init = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))

        di = get_ac_distance_time(s_init, dx_1, dx_2, dy_1, dy_2, 1)

        return s_init - di


def get_delta_dst_t(lat_1, lon_1, lat_2, lon_2, hdg_1,
                    hdg_2, spd_1, spd_2, t):

    _gamma, _alpha, _beta = gamma_angle(lat_1, lon_1, lat_2, lon_2, hdg_1,
                                        hdg_2)

    if np.isnan(_gamma):
        return np.nan
    else:
        dx_1, dx_2, dy_1, dy_2 = get_dx_dy(_alpha, _beta, spd_1, spd_2)
        s_init = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))

        di = get_ac_distance_time(s_init, dx_1, dx_2, dy_1, dy_2, t)

        return abs(s_init - di)


def get_ttc_est(lat_1, lon_1, lat_2, lon_2, hdg_1,
                hdg_2, spd_1, spd_2, _tmax):

    _gamma, _alpha, _beta = gamma_angle(lat_1, lon_1, lat_2, lon_2, hdg_1,
                                        hdg_2)

    if np.isnan(_gamma):
        return np.nan
    else:
        dx_1, dx_2, dy_1, dy_2 = get_dx_dy(_alpha, _beta, spd_1, spd_2)
        s_init = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))

        for t in range(_tmax):
            di = get_ac_distance_time(s_init, dx_1, dx_2, dy_1, dy_2, t)
            if di < ipz_range:
                return t

        return np.nan

#
# def get_intent_ttc_est(lat_1, lon_1, lat_2, lon_2, hdg_1,
#                        hdg_2, spd_1, spd_2, _tmax, _cps_1, _cps_2):
#
#     dst_cp_1 = _cps_1[0][0] * spd_1 * knts_ms_ratio
#     dst_cp_2 = _cps_2[0][0] * spd_2 * knts_ms_ratio
#     lat_1_new = copy(lat_1)
#     lon_1_new = copy(lon_1)
#     lat_2_new = copy(lat_2)
#     lon_2_new = copy(lon_2)
#     hdg_1_new = copy(hdg_1)
#     hdg_2_new = copy(hdg_2)
#
#     for t in range(_tmax):
#
#         t_cp_1 = [c[0] for c in _cps_1 if t >= c[0]][0]
#         t_cp_2 = [c[0] for c in _cps_2 if t >= c[0]][0]
#         hdg_cp_1 = [c[1] for c in _cps_1 if t >= c[0]][0]
#         hdg_cp_2 = [c[1] for c in _cps_2 if t >= c[0]][0]
#
#         if t >= t_cp_1:
#             (lat_1_new, lon_1_new) = find_coord_dst_hdg((lat_1, lon_1),
#                                                         hdg_1, dst_cp_1)
#             hdg_1_new = hdg_cp_1
#
#         if t >= t_cp_2:
#             (lat_2_new, lon_2_new) = find_coord_dst_hdg((lat_2, lon_2),
#                                                         hdg_2, dst_cp_2)
#             hdg_2_new = hdg_cp_2
#
#         _gamma, _alpha, _beta = gamma_angle(lat_1_new, lon_1_new, lat_2_new,
#                                             lon_2_new, hdg_1_new, hdg_2_new)
#
#         if np.isnan(_gamma):
#             continue
#
#         dx_1, dx_2, dy_1, dy_2 = get_dx_dy(_alpha, _beta, spd_1,
#                                            spd_2)
#
#         s_init = calc_coord_dst((lat_1_new, lon_1_new),
#                                 (lat_2_new, lon_2_new))
#
#         di = get_ac_distance_time(s_init, dx_1, dx_2, dy_1, dy_2, t)
#
#         if di < ipz_range:
#             return t
#
#     return np.nan


def get_next_crd(lat, lon, hdg, spd):

    dst = spd * knts_ms_ratio * 1
    (lat_n, lon_n) = find_coord_dst_hdg((lat, lon), hdg, dst)

    return lat_n, lon_n


def hdg_diff_next_wp(lat, lon, wpc, wpn):
    hdg_ac_wpn = calc_compass_bearing((lat, lon), wpn)
    hdg_wpc_wpn = calc_compass_bearing(wpc, wpn)

    hdiff = heading_diff(hdg_ac_wpn, hdg_wpc_wpn)

    return hdiff


def sequentialize(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_intent_ttc_est(lat_1, lon_1, lat_2, lon_2, hdg_1,
                       hdg_2, spd_1, spd_2, _tmax, _wps_1, _wps_2):

    wps_seq_1 = sequentialize(_wps_1)
    wps_seq_2 = sequentialize(_wps_2)

    lat_n_1, lon_n_1, lat_n_2, lon_n_2 = lat_1, lon_1, lat_2, lon_2

    try:

        try:

            if not len(wps_seq_1) < 2:
                wp1_curr_iter = iter(wps_seq_1)
                wp1_next_iter = iter(wps_seq_1[1:])
                _wp1_curr = next(wp1_curr_iter, wps_seq_1[-2])
                _wp1_next = next(wp1_next_iter, wps_seq_1[-1])

            else:
                wp1_curr_iter, wp1_next_iter, _wp1_curr, _wp1_next = \
                    None, None, None, None

            if not len(wps_seq_2) < 2:
                wp2_curr_iter = iter(wps_seq_2)
                wp2_next_iter = iter(wps_seq_2[1:])
                _wp2_curr = next(wp2_curr_iter, wps_seq_2[-2])
                _wp2_next = next(wp2_next_iter, wps_seq_2[-1])

            else:
                wp2_curr_iter, wp2_next_iter, _wp2_curr, _wp2_next = \
                    None, None, None, None

        except Exception as e1:
            print('wp init failed')
            print(e1)

        for t in range(1, _tmax):

            if t == 1:
                lat_n_1, lon_n_1 = get_next_crd(lat_1, lon_1, hdg_1, spd_1)
                lat_n_2, lon_n_2 = get_next_crd(lat_2, lon_2, hdg_2, spd_2)

            else:
                lat_n_1, lon_n_1 = get_next_crd(lat_n_1, lon_n_1, hdg_1, spd_1)
                lat_n_2, lon_n_2 = get_next_crd(lat_n_2, lon_n_2, hdg_2, spd_2)

            di = calc_coord_dst((lat_n_1, lon_n_1), (lat_n_2, lon_n_2))

            if di < ipz_range:
                return t

            if not len(wps_seq_1) < 2:
                _hdiff1 = hdg_diff_next_wp(lat_n_1, lon_n_1,
                                           _wp1_curr, _wp1_next)
                dst_currwp_1 = calc_coord_dst((lat_n_1, lon_n_1), _wp1_curr)

                if abs(_hdiff1) < 5 and dst_currwp_1 < 8000:
                    hdg_1 = calc_compass_bearing(_wp1_curr, _wp1_next)
                    _wp1_curr = next(wp1_curr_iter, sequentialize(_wps_1)[-2])
                    _wp1_next = next(wp1_next_iter, sequentialize(_wps_1)[-1])


            if not len(wps_seq_2) < 2:
                _hdiff2 = hdg_diff_next_wp(lat_n_2, lon_n_2,
                                           _wp2_curr, _wp2_next)

                dst_currwp_2 = calc_coord_dst((lat_n_2, lon_n_2), _wp2_curr)

                if abs(_hdiff2) < 5 and dst_currwp_2 < 8000:
                    hdg_2 = calc_compass_bearing(_wp2_curr, _wp2_next)
                    _wp2_curr = next(wp2_curr_iter, sequentialize(_wps_2)[-2])
                    _wp2_next = next(wp2_next_iter, sequentialize(_wps_2)[-1])


        return np.nan

    except Exception as e:
        print('intent ttc func failed')
        print(e)
        return np.nan


def get_proba_ttc_est(lat_1, lon_1, lat_2, lon_2, hdg_1, hdg_2, spd_1,
                      spd_2, pct, i_kde_dict, _tmax):

    assert str(pct) in list(i_kde_dict.keys()), "Percentage not in dict"

    _gamma, _alpha, _beta = gamma_angle(lat_1, lon_1, lat_2, lon_2, hdg_1,
                                        hdg_2)

    kde_bounds = (i_kde_dict[str(pct)], i_kde_dict[str(100-pct)])

    if np.isnan(_gamma):
        return np.nan
    else:
        dx_1, dx_2, dy_1, dy_2 = get_dx_dy(_alpha, _beta, spd_1, spd_2)
        s_init = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))

        for t in range(_tmax):
            di = get_ac_distance_time(s_init, dx_1, dx_2, dy_1, dy_2, t)
            if any((di + b*di*t) < ipz_range for b in kde_bounds):
                return t

        return np.nan


def crop_ddr2_flight(fl):

    first_ddr2_wp_1 = find_waypoint_index_v2((fl['lat'][0], fl['lon'][0]),
                                             list(zip(fl['lat_seg_b'],
                                                      fl['lon_seg_b'])))
    last_ddr2_wp_1 = find_waypoint_index_v2((fl['lat'][-1], fl['lon'][-1]),
                                            list(zip(fl['lat_seg_b'],
                                                     fl['lon_seg_b'])))

    for kd in ['ep_seg_b', 'lat_seg_b', 'lon_seg_b',
               'fl_seg_b', 'fl_seg_e', 'seq']:

        fl[kd] = fl[kd][first_ddr2_wp_1:last_ddr2_wp_1 + 2]

    return fl


def crop_ddr2_flight_seg(fl):

    first_ddr2_seg = find_wp_segment((fl['lat'][0], fl['lon'][0]),
                                     list(zip(fl['lat_seg_b'],
                                              fl['lon_seg_b'],
                                              fl['lat_seg_e'],
                                              fl['lon_seg_e'])))

    last_ddr2_seg = find_wp_segment((fl['lat'][-1], fl['lon'][-1]),
                                    list(zip(fl['lat_seg_b'],
                                             fl['lon_seg_b'],
                                             fl['lat_seg_e'],
                                             fl['lon_seg_e'])))

    if not all(i for i in [first_ddr2_seg, last_ddr2_seg]):
        return None

    for kd in ['ep_seg_b', 'lat_seg_b', 'lon_seg_b', 'lat_seg_e', 'lon_seg_e',
               'fl_seg_b', 'fl_seg_e', 'seq']:

        fl[kd] = fl[kd][first_ddr2_seg[0]:last_ddr2_seg[0]+2]

    return fl


def realign_conflict(b):
    cfl_dst = ipz_range

    if len(b['lon_1']) != len(b['lon_2']):
        print('Flights not the same length')
        return None

    for i in range(1, len(b)):

        if calc_coord_dst((b['lon_1'][-i], b['lat_1'][-i]),
                          (b['lon_2'][-i], b['lat_2'][-i])) >= cfl_dst:

            for k in ['ts_1', 'lat_1', 'lon_1', 'alt_1', 'spd_1', 'hdg_1',
                      'roc_1', 'ts_2', 'lat_2', 'lon_2', 'alt_2', 'spd_2',
                      'hdg_2', 'roc_2']:

                b[k] = b[k][:-(i - 1)]

            return b

    return None


def confl_check(lat_1, lon_1, lat_2, lon_2):

    d = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))

    if d < ipz_range:
        return 1
    else:
        return 0


def time_to_conflict(tr1, tr2):

    for i in range(len(tr1)):
        cdst = calc_coord_dst((tr1['proj_lat'].iloc[i],
                              tr1['proj_lon'].iloc[i]),
                              (tr2['proj_lat'].iloc[i],
                              tr2['proj_lon'].iloc[i]))

        if cdst <= ipz_range:
            ttc = tr1['ts'].iloc[i] - tr1['ts'].iloc[0]

            return ttc

    return None


def get_detection_class(ttc_list):

    confl_class = []

    for ttce, ttca in ttc_list:

        if np.isnan(ttce) and np.isnan(ttca):
            confl_class.append('TN')
            continue

        if np.isnan(ttce) and ~np.isnan(ttca):
            confl_class.append('FN')
            continue

        if ~np.isnan(ttce) and np.isnan(ttca):
            confl_class.append('FP')
            continue

        if ~np.isnan(ttce) and ~np.isnan(ttca) and (abs(ttce - ttca) <=
                                                    t_overshoot_max):
            confl_class.append('TP')
            continue

        if ~np.isnan(ttce) and ~np.isnan(ttca) and (abs(ttce - ttca) >
                                                    t_overshoot_max):
            confl_class.append('FN')
            continue

        continue

    return confl_class


def get_detection_class_single(ttca, ttce):

    if np.isnan(ttce) and np.isnan(ttca):
        return 'TN'

    if np.isnan(ttce) and ~np.isnan(ttca) and ttca <= la_time:
        return 'FN'

    if ~np.isnan(ttce) and np.isnan(ttca):
        return 'FP'

    if ~np.isnan(ttce) and ~np.isnan(ttca) and ttca <= la_time\
            and (-120 < (ttce - ttca) <= ttca*0.1):
        return 'TP'

    if ~np.isnan(ttce) and ~np.isnan(ttca) and ttca <= la_time \
            and ((ttce - ttca) < -120):
        return 'FP'

    if ~np.isnan(ttce) and ~np.isnan(ttca) and ttca <= la_time\
            and ((ttce - ttca) > ttca*0.1):
        return 'FN'

    return 'Error'


def create_performance_dict(ttc_list):

    class_dict = {'TP': [], 'FP': [], 'TN': [], 'FN': []}

    ttc_cl = get_detection_class(ttc_list)

    class_dict['TP'].append(ttc_cl.count('TP'))
    class_dict['FP'].append(ttc_cl.count('FP'))
    class_dict['TN'].append(ttc_cl.count('TN'))
    class_dict['FN'].append(ttc_cl.count('FN'))

    return class_dict


def f1_score(tp, fp, tn, fn):

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    f1 = 2 * (1 / (1 / prec + 1 / rec))

    return f1


def create_la_performance_dict(ttc_list, bin_sec=20):

    class_list = []
    bin_dp_df = {}

    for v in ttc_list:
        class_list.append((v[0], get_detection_class_single(v[1], v[2])))

    class_list = sorted(class_list, key=lambda tup: tup[0])

    for b in range(int(la_time / bin_sec)):
        bmin = b * bin_sec
        bmax = bmin + bin_sec

        ri = [v[1] for v in class_list if bmin <= v[0] < bmax]

        bin_dp_df[str(bmax)] = {'TP': ri.count('TP'), 'FP': ri.count('FP'),
                                'TN': ri.count('TN'), 'FN': ri.count('FN')}

    return bin_dp_df


def get_tcps(b):

    _cps_1 = []
    _cps_2 = []

    for i, t in enumerate(b['ts_1']):

        _hdg_cp_1 = b['hdg_e_1'][i]
        _crd_cp_1 = (b['lat_1'][i], b['lon_1'][i])

        _hdg_cp_2 = b['hdg_e_2'][i]
        _crd_cp_2 = (b['lat_2'][i], b['lon_2'][i])

        _next_wp_hdgs_1 = [h for h in b['hdg_e_1'][i:]
                           if h != b['hdg_e_1'][i]]

        _next_wp_hdgs_2 = [h for h in b['hdg_e_2'][i:]
                           if h != b['hdg_e_2'][i]]

        if len(_next_wp_hdgs_1) > 0:
            _hdg_cp_1 = _next_wp_hdgs_1[0]
            _crd_cp_1 = [(c[0], c[1]) for c in
                         zip(b['lat_1'][i:], b['lon_1'][i:],
                             b['hdg_e_1'][i:])
                         if c[2] == _hdg_cp_1][0]

        if len(_next_wp_hdgs_2) > 0:
            _hdg_cp_2 = _next_wp_hdgs_2[0]
            _crd_cp_2 = [(c[0], c[1]) for c in
                         zip(b['lat_2'][i:], b['lon_2'][i:],
                             b['hdg_e_2'][i:])
                         if c[2] == _hdg_cp_2][0]

        _dst_cp_1 = calc_coord_dst(_crd_cp_1, (b['lat_1'][i],
                                               b['lon_1'][i]))

        _dst_cp_2 = calc_coord_dst(_crd_cp_2, (b['lat_2'][i],
                                               b['lon_2'][i]))

        _t_cp_1 = _dst_cp_1 / (b['spd_e_1'][i] * knts_ms_ratio)
        _t_cp_2 = _dst_cp_2 / (b['spd_e_2'][i] * knts_ms_ratio)

        _cps_1.append((_t_cp_1, _hdg_cp_1))
        _cps_2.append((_t_cp_2, _hdg_cp_2))

    return _cps_1, _cps_2


def process_flight_batch_tstep(bx, prob_flag=False, intent_flag=False):

    exp_keys = ['ts_1', 'ts_2', 'lat_1', 'lon_1', 'lat_2', 'lon_2', 'hdg_1',
                'hdg_2', 'hdg_e_1', 'hdg_e_2', 'spd_1', 'spd_2', 'spd_e_1',
                'spd_e_2']

    if intent_flag is True:
        exp_keys.extend(['wp_seg_1', 'wp_seg_2'])

    ttc_res = []

    _pct_dict = {}
    pcti = None

    confl_count = 0
    non_confl_count = 0

    if prob_flag:
        _pct_dict = load_obj("../tools/kde_dictionary.pkl")
        pcti = 35

    for bi in bx:

        ttc_res_i = []
        b = {}

        for k in exp_keys:
            try:
                b[k] = bi[k]
            except Exception as e:
                print(e)
                print('Not all keys present in flight dictionary')
                continue

        confl_flag = []

        try:
            for i in range(len(b['ts_1'])):
                confl_flag.append(confl_check(b['lat_1'][i], b['lon_1'][i],
                                              b['lat_2'][i], b['lon_2'][i]))

            confl_ix = np.where(np.array(confl_flag) > 0)[0]

            if len(confl_ix) > 0:
                confl_i = confl_ix[0]
                confl_ts = b['ts_1'][confl_i]

            else:
                confl_ts = None

            if confl_ts:
                print('conflict')
                tmax = confl_ts
                confl_count = confl_count + 1
            else:
                tmax = b['ts_1'][-1]
                non_confl_count = non_confl_count + 1

        except Exception as e:
            print('Creating ttc_act failed:')
            print(e)
            continue

        try:

            for i, t in enumerate(b['ts_1']):

                if t > tmax:
                    break

                if confl_ts:
                    ttc_act_i = confl_ts - b['ts_1'][i]
                    t_la_max = int((tmax - b['ts_1'][i]) * 2)
                else:
                    ttc_act_i = np.nan
                    t_la_max = int(tmax - b['ts_1'][i])

                if prob_flag is True:
                    ttc_est_i = get_proba_ttc_est(b['lat_1'][i],
                                                  b['lon_1'][i],
                                                  b['lat_2'][i],
                                                  b['lon_2'][i],
                                                  b['hdg_e_1'][i],
                                                  b['hdg_e_2'][i],
                                                  b['spd_e_1'][i],
                                                  b['spd_e_2'][i],
                                                  pcti, _pct_dict,
                                                  t_la_max)

                elif intent_flag is True:

                    wp_list_1 = [(wp[2][2], wp[2][3]) for wp in
                                 b['wp_seg_1'][i:]]
                    wp_list_2 = [(wp[2][2], wp[2][3]) for wp in
                                 b['wp_seg_2'][i:]]

                    if not wp_list_1:
                        wp_list_1 = [(b['wp_seg_1'][-1][2][2],
                                      b['wp_seg_1'][-1][2][3])]
                    if not wp_list_2:
                        wp_list_2 = [(b['wp_seg_2'][-1][2][2],
                                      b['wp_seg_2'][-1][2][3])]

                    try:
                        ttc_est_i = get_intent_ttc_est(b['lat_1'][i],
                                                       b['lon_1'][i],
                                                       b['lat_2'][i],
                                                       b['lon_2'][i],
                                                       b['hdg_e_1'][i],
                                                       b['hdg_e_2'][i],
                                                       b['spd_e_1'][i],
                                                       b['spd_e_2'][i],
                                                       t_la_max, wp_list_1,
                                                       wp_list_2)
                    except Exception as e:
                        print('Intent TTC failed')
                        print(e)
                        continue

                else:
                    ttc_est_i = get_ttc_est(b['lat_1'][i], b['lon_1'][i],
                                            b['lat_2'][i], b['lon_2'][i],
                                            b['hdg_e_1'][i],
                                            b['hdg_e_2'][i],
                                            b['spd_e_1'][i],
                                            b['spd_e_2'][i], t_la_max)

                # if ttc_est_i:
                ttc_res_i.append((t_la_max, ttc_act_i, ttc_est_i))

        except Exception as e:
            print('Creating ttc_est failed:')
            print(e)
            print(i)
            print(len(b['ts_1']))
            continue

        ttc_res.extend(ttc_res_i)

    print('Conflicts: %d' % confl_count)
    print('Non Conflicts: %d' % non_confl_count)

    return ttc_res


# def process_proba_flight_batch(bc):
#
#     return process_flight_batch(bc, prob_flag=True)


def process_proba_flight_batch_tstep(bc):

    return process_flight_batch_tstep(bc, prob_flag=True)


def process_intent_flight_batch_tstep(bc):

    return process_flight_batch_tstep(bc, intent_flag=True)


def preprocess_det_conflicts(batch):
    res_batch = []

    for b in batch:
        b['hdg_e_1'] = b['hdg_1']
        b['hdg_e_2'] = b['hdg_2']
        b['spd_e_1'] = b['spd_1']
        b['spd_e_2'] = b['spd_2']

        b_tlen = max(b['ts_1']) - min(b['ts_1'])

        if b_tlen >= la_time:
            res_batch.append(b)

    return res_batch


def preprocess_intent_conflicts(batch):
    res_batch = []

    for bi in batch:

        b = {}

        if bi:

            b_tlen = max(bi['ts_1']) - min(bi['ts_1'])

            if b_tlen < la_time:
                continue

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
                    # fl1 = fl1[fl1.wp_seg.notnull()]
                except Exception as e:
                    print('Adding waypoints to flight failed, error: ')
                    print(e)
                    continue

                if fl1 is None:
                    continue

                if len(fl1) == 0:
                    continue

                fl2 = {}

                for k in fl_keys:
                    fl2[k] = bi["%s%s" % (k, '_2')]

                try:
                    fl2 = crop_ddr2_flight_seg(fl2)
                except Exception as e:
                    print('Cropping DDR2 flight failed, error: ')
                    print(e)
                    continue

                if fl2 is None:
                    continue

                try:
                    fl2 = add_waypoint_segments(fl2)
                    # fl2 = fl2[fl2.wp_seg.notnull()]
                except Exception as e:
                    print('Adding waypoints to flight failed, error: ')
                    print(e)
                    continue

                if fl2 is None:
                    continue

                if len(fl2) == 0:
                    continue

                fl2_n = fl2[fl2.wp_seg.notnull() & fl1.wp_seg.notnull()]
                fl1_n = fl1[fl2.wp_seg.notnull() & fl1.wp_seg.notnull()]

                fl2_n = fl2_n.reset_index(drop=True)
                fl1_n = fl1_n.reset_index(drop=True)

                if len(fl2_n) == 0 or len(fl1_n) == 0:
                    continue

                fl1_n['hdg_int'] = fl1_n.apply(lambda x: calc_compass_bearing(
                    (x['wp_seg'][2][0], x['wp_seg'][2][1]),
                    (x['wp_seg'][2][2], x['wp_seg'][2][3])),
                                           axis=1)

                fl2_n['hdg_int'] = fl2_n.apply(lambda x: calc_compass_bearing(
                    (x['wp_seg'][2][0], x['wp_seg'][2][1]),
                    (x['wp_seg'][2][2], x['wp_seg'][2][3])),
                                           axis=1)

                for k in ['ts', 'lat', 'lon', 'hdg', 'spd', 'wp_seg']:
                    b['%s_1' % k] = fl1_n[k].tolist()
                    b['%s_2' % k] = fl2_n[k].tolist()

                b['hdg_e_1'] = fl1_n['hdg_int'].tolist()
                b['hdg_e_2'] = fl2_n['hdg_int'].tolist()
                b['spd_e_1'] = fl1_n['spd'].tolist()
                b['spd_e_2'] = fl2_n['spd'].tolist()

                res_batch.append(b)

            except Exception as e:
                print('Preprocessing data failed, error:')
                print(e)
                continue

    return res_batch


# def preprocess_intent_conflicts(batch):
#     res_batch = []
#
#     for bi in batch:
#
#         b = {}
#
#         if bi:
#             fl_keys = ['ts', 'lat', 'lon', 'hdg', 'alt', 'spd',
#                        'roc', 'ep_seg_b', 'lat_seg_b', 'lon_seg_b',
#                        'fl_seg_b', 'fl_seg_e', 'seq']
#
#             try:
#                 fl1 = {}
#                 for k in fl_keys:
#                     fl1[k] = bi["%s%s" % (k, '_1')]
#
#                 try:
#                     fl1 = crop_ddr2_flight(fl1)
#                 except Exception as e:
#                     print('Cropping DDR2 flight failed, error: ')
#                     print(e)
#                     continue
#
#                 try:
#                     fl1 = add_waypoints(fl1)
#                 except Exception as e:
#                     print('Adding waypoints to flight failed, error: ')
#                     print(e)
#                     continue
#
#                 if fl1 is None:
#                     continue
#
#                 if len(fl1) == 0:
#                     continue
#
#                 fl1['hdg_int'] = fl1.apply(lambda x:
#                                            calc_compass_bearing(x['curr_wp'],
#                                                                 x['next_wp']),
#                                            axis=1)
#
#                 fl2 = {}
#
#                 for k in fl_keys:
#                     fl2[k] = bi["%s%s" % (k, '_2')]
#
#                 try:
#                     fl2 = crop_ddr2_flight(fl2)
#                 except Exception as e:
#                     print('Cropping DDR2 flight failed, error: ')
#                     print(e)
#                     continue
#
#                 try:
#                     fl2 = add_waypoints(fl2)
#                 except Exception as e:
#                     print('Adding waypoints to flight failed, error: ')
#                     print(e)
#                     continue
#
#                 if fl2 is None:
#                     continue
#
#                 if len(fl2) == 0:
#                     continue
#
#                 fl2['hdg_int'] = fl2.apply(lambda x:
#                                            calc_compass_bearing(x['curr_wp'],
#                                                                 x['next_wp']),
#                                            axis=1)
#
#                 for k in ['ts', 'lat', 'lon', 'hdg', 'spd']:
#                     b['%s_1' % k] = fl1[k]
#                     b['%s_2' % k] = fl2[k]
#
#                 b['hdg_e_1'] = fl1['hdg_int']
#                 b['hdg_e_2'] = fl2['hdg_int']
#                 b['spd_e_1'] = fl1['spd']
#                 b['spd_e_2'] = fl2['spd']
#
#                 res_batch.append(b)
#
#             except Exception as e:
#                 print('Preprocessing data failed, error:')
#                 print(e)
#                 continue
#
#     return res_batch

# def process_flight_batch(bx, prob_flag=False):
#
#     exp_keys = ['ts_1', 'ts_2', 'lat_1', 'lon_1', 'lat_2', 'lon_2', 'hdg_1',
#                 'hdg_2', 'hdg_e_1', 'hdg_e_2', 'spd_1', 'spd_2', 'spd_e_1',
#                 'spd_e_2']
#
#     ttca_bin = []
#     ttce_bin = []
#     ttc_est = []
#     ttc_act = []
#
#     _pct_dict = {}
#     pcti = None
#
#     if prob_flag:
#         _pct_dict = load_obj("../tools/kde_dictionary.pkl")
#         pcti = 30
#
#     for bi in bx:
#
#         ttc_est_i = []
#         b = {}
#
#         for k in exp_keys:
#             try:
#                 b[k] = bi[k]
#             except Exception as e:
#                 print(e)
#                 print('Not all keys present in flight dictionary')
#                 continue
#
#         confl_flag = []
#
#         try:
#             for i in range(len(b['ts_1'])):
#                 confl_flag.append(confl_check(b['lat_1'][i], b['lon_1'][i],
#                                               b['lat_2'][i], b['lon_2'][i]))
#
#             confl_ix = np.where(np.array(confl_flag) > 0)[0]
#
#             if len(confl_ix) > 0:
#                 confl_i = confl_ix[0]
#                 confl_ts = b['ts_1'][confl_i]
#                 ttc_act_i = [confl_ts - t for t in b['ts_1']]
#
#             else:
#                 confl_ts = None
#                 ttc_act_i = [np.nan] * len(b['ts_1'])
#
#             if confl_ts:
#                 tmax = confl_ts + t_overshoot_max
#             else:
#                 tmax = b['ts_1'][-1] + t_overshoot_max
#
#         except Exception as e:
#             print('Creating ttc_act failed:')
#             print(e)
#             continue
#
#         try:
#             if prob_flag:
#
#                 for i in range(len(b['ts_1'])):
#                     t_la_max = min(int(tmax - b['ts_1'][i]), la_time)
#                     ttc_est_i.append(get_proba_ttc_est(b['lat_1'][i],
#                                                        b['lon_1'][i],
#                                                        b['lat_2'][i],
#                                                        b['lon_2'][i],
#                                                        b['hdg_e_1'][i],
#                                                        b['hdg_e_2'][i],
#                                                        b['spd_e_1'][i],
#                                                        b['spd_e_2'][i],
#                                                        pcti, _pct_dict,
#                                                        t_la_max))
#
#             else:
#                 for i in range(len(b['ts_1'])):
#                     t_la_max = min(int(tmax - b['ts_1'][i]), la_time)
#                     ttc_est_i.append(get_ttc_est(b['lat_1'][i], b['lon_1'][i],
#                                                  b['lat_2'][i], b['lon_2'][i],
#                                                  b['hdg_e_1'][i],
#                                                  b['hdg_e_2'][i],
#                                                  b['spd_e_1'][i],
#                                                  b['spd_e_2'][i], t_la_max))
#
#         except Exception as e:
#             print('Creating ttc_est failed:')
#             print(e)
#             continue
#
#         ttc_est.extend(ttc_est_i)
#         ttc_act.extend(ttc_act_i)
#         ttca_bin.extend([1 if ~np.isnan(t) else 0 for t in ttc_act])
#         ttce_bin.extend([1 if ~np.isnan(t) else 0 for t in ttc_est])
#
#     return list(zip(ttc_est, ttc_act))
