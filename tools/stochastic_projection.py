
import numpy as np
import math
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from tools.flight_projection import calc_coord_dst, calc_compass_bearing, \
                                    heading_diff
from conf.config import ipz_range, knts_ms_ratio


def mc_estimate_distr(x_obs):
    x_obs_p = np.nanpercentile(x_obs, 99)
    x_obs_filt = [v for v in x_obs if abs(v) < x_obs_p]

    rlim = x_obs_p

    with pm.Model() as model:
        b = pm.Uniform('b', 0, 5000)
        a = pm.Uniform('a', -5000, 5000)
        cte = pm.Cauchy('cte', alpha=a, beta=b, observed=x_obs)

        x_trace = pm.sample(500, tune=2000)

    a_est = np.mean(x_trace['a'])
    b_est = np.mean(x_trace['b'])

    return a_est, b_est


def get_distribution_params(prefix, obs_df):
    assert prefix in ['_cte', '_ate'], "Prefix not correct"

    alpha_est = []
    beta_est = []
    x_la = []

    for k in obs_df.keys():
        if prefix in k:
            x_obs = [x for x in obs_df[k] if ~np.isnan(x)]
            range_lim = np.nanpercentile(x_obs, 99)
            x_obs_filt = [x for x in x_obs if abs(x) < range_lim][:2000]

            x_lai = int(k.strip(prefix))
            x_la.append(x_lai)

            print('Estimating for t=%d' % x_lai)

            a_est, b_est = mc_estimate_distr(x_obs_filt)

            alpha_est.append(a_est)
            beta_est.append(b_est)

    from sklearn import linear_model

    x_in = [[1, xx, xx ** 2] for xx in x_la]
    clf_dict = {}
    param_dict = {}
    param_dict['la_time'] = x_la
    param_dict['alpha'] = alpha_est
    param_dict['beta'] = beta_est

    for param in ['alpha', 'beta']:
        clf_dict[param] = linear_model.LinearRegression()
        clf_dict[param].fit(x_in, param_dict[param])

    return param_dict, clf_dict


def plot_est_distr(la_time, x_obs_i, clf_dict):
    x_obs_p_i = np.nanpercentile(x_obs_i, 99)
    x_obs_filt_i = [v for v in x_obs_i if abs(v) < x_obs_p_i]

    rlim = x_obs_p_i

    a_est_i = clf_dict['alpha'].predict([[1, la_time, la_time ** 2]])
    b_est_i = clf_dict['beta'].predict([[1, la_time, la_time ** 2]])

    sdistr = pm.Cauchy.dist(alpha=a_est_i, beta=b_est_i)
    x_samp = [v for v in sdistr.random(size=100000) if abs(v) < rlim]

    fig, ax = plt.subplots(figsize=(15, 10))
    n_bins = 1000

    n, bins, patches = ax.hist(x_samp, n_bins,
                               normed=0,
                               histtype='stepfilled',
                               cumulative=False,
                               label='CTE distr. at %s seconds' % la_time,
                               alpha=0.6)

    n, bins, patches = ax.hist(x_obs_filt_i, n_bins,
                               normed=0,
                               histtype='stepfilled',
                               cumulative=False,
                               label='CTE distr. at %s seconds' % la_time,
                               alpha=0.6)
    plt.xlim(-rlim, rlim)
    plt.show()


def ac_dist_stoch(t, cte_1, ate_1, cte_2, ate_2, lat_1, lon_1, lat_2, lon_2,
                  hdg_1, hdg_2, spd_1, spd_2):

    knots_to_ms = knts_ms_ratio
    ipz_lim = ipz_range  # meters

    alpha_1 = heading_diff(calc_compass_bearing((lat_1, lon_1),
                                                (lat_2, lon_2)), hdg_1)

    alpha_2 = heading_diff(calc_compass_bearing((lat_2, lon_2),
                                                (lat_1, lon_1)), hdg_2)

    gamma = 180 - (alpha_1 + alpha_2)

    if gamma < 0:
        return np.nan
    else:
        dx1_e = math.sin(math.radians(heading_diff(0, hdg_1))) * ate_1 + \
            math.cos(math.radians(heading_diff(0, hdg_1))) * cte_1

        dy1_e = math.sin(math.radians(heading_diff(hdg_1, 0))) * cte_1 + \
            math.cos(math.radians(heading_diff(hdg_1, 0))) * ate_1

        dx2_e = math.sin(math.radians(heading_diff(0, hdg_2))) * ate_2 + \
            math.cos(math.radians(heading_diff(0, hdg_2))) * cte_2

        dy2_e = math.sin(math.radians(heading_diff(hdg_2, 0))) * cte_2 + \
            math.cos(math.radians(heading_diff(hdg_2, 0))) * ate_2

        dy_1 = math.cos(math.radians(heading_diff(hdg_1, 0))) * spd_1 * \
            knots_to_ms + (dy1_e / t)

        dy_2 = math.cos(math.radians(heading_diff(hdg_2, 0))) * spd_2 * \
            knots_to_ms + (dy2_e / t)

        dx_1 = math.sin(math.radians(heading_diff(hdg_1, 0))) * spd_1 * \
            knots_to_ms + (dx1_e / t)

        dx_2 = math.sin(math.radians(heading_diff(hdg_2, 0))) * spd_2 * \
            knots_to_ms + (dx2_e / t)

        dx = abs(dx_1 - dx_2) * t
        dy = abs(dy_1 - dy_2) * t

        s = calc_coord_dst((lat_1, lon_1), (lat_2, lon_2))
        d = s - np.sqrt((dy ** 2 + dx ** 2))

        return d


def create_kde(df, la_t, gsize=100j):

    cte_data_raw = [x for x in df['%d_cte' % la_t] if ~np.isnan(x)]
    cte_data_p = np.percentile(cte_data_raw, 99)
    cte_data_filt = [v for v in cte_data_raw if abs(v) < cte_data_p][:2000]
    ate_data_raw = [x for x in df['%d_ate' % la_t] if ~np.isnan(x)]
    ate_data_p = np.percentile(ate_data_raw, 99)
    ate_data_filt = [v for v in ate_data_raw if abs(v) < ate_data_p][:2000]

    data = np.vstack([cte_data_filt, ate_data_filt])
    # X, Y = np.mgrid[min(cte_data_filt):max(cte_data_filt):gsize,
    # min(ate_data_filt):max(ate_data_filt):gsize]
    # grid = np.vstack([Y.ravel(), X.ravel()])

    kde = gaussian_kde(data, bw_method=(20 / np.std(data)))
    #     pdf = kde(grid)

    return kde
