import sys
sys.path.append("..")

from scipy.stats import gaussian_kde
from psycopg2.extras import RealDictCursor
from tools.flight_projection import *
from tools.db_connector import get_pg_conn
from tools.conflict_handling import get_delta_dst, get_delta_dst_t
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as filename:
        pickle.dump(obj, filename, pickle.HIGHEST_PROTOCOL)


def sample_dst_kde(kde, hdg, dst_space, pct, gs):

    pdf = kde(np.array([[hdg]*gs, [v for v in dst_space]]))
    pdf_zip = list(zip(dst_space, np.cumsum(pdf) / max(np.cumsum(pdf))))

    pct_low = [(d, p) for (d, p) in pdf_zip if p <= 0.5 - pct][-1][0]
    pct_high = [(d, p) for (d, p) in pdf_zip if p <= 0.5 + pct][-1][0]

    return pct_low, pct_high


def distr_func(_d, _dct, _t):
    return (_d - _dct) / (_dct * _t)


# Get database connection and define global variables

conn = get_pg_conn()
cur_read = conn.cursor(cursor_factory=RealDictCursor)
cur_read.execute("SELECT ts_1, ts_2, lat_1, lon_1, lat_2, lon_2, \
                    hdg_1, hdg_2, spd_1, spd_2 \
                    FROM public.overlapping_flights limit 5000;")
batch = cur_read.fetchall()

res_batch = []
bin_dict = {}
pdf_dict = {}

la_time = 1200

print('Preprocessing flights')

for b in batch:

    if len(b['ts_1']) != len(b['ts_2']):
        print('Flights do not have same lengths')
        continue

    f_df = pd.DataFrame.from_dict(b)
    if len(f_df) > 0:

        dstc_lst = []
        dst_lst = []

        for i, r in f_df.iterrows():
            dstc_lst.append(
                get_delta_dst(r['lat_1'], r['lon_1'], r['lat_2'], r['lon_2'],
                              r['hdg_1'], r['hdg_2'], r['spd_1'], r['spd_2']))

            dst_lst.append(calc_coord_dst((r['lat_1'], r['lon_1']),
                                          (r['lat_2'], r['lon_2'])))

        f_df['dst_change_i'] = dstc_lst
        f_df['dst'] = dst_lst

        f_df = f_df.dropna(how='any')
        f_df = f_df.reset_index(drop=True)

        if len(f_df) > 0:
            f_df['dst_change'] = [f_df['dst_change_i'].iloc[0]] * len(f_df)
            f_df['dst_diff'] = f_df['dst'].iloc[0] - f_df['dst']
            f_df['la_t'] = f_df['ts_1'] - f_df['ts_1'].iloc[0]

            delta_dst_t = []

            for i, rr in f_df.iterrows():
                delta_dst_t.append(
                    get_delta_dst_t(rr['lat_1'], rr['lon_1'], rr['lat_2'],
                                    rr['lon_2'],
                                    rr['hdg_1'], rr['hdg_2'], rr['spd_1'],
                                    rr['spd_2'], rr['la_t']))

            f_df['delta_dst_t'] = delta_dst_t

            f_df = f_df.dropna(how='any')
            f_df = f_df.reset_index(drop=True)

            if len(f_df) > 0:
                res_batch.append(f_df.to_dict(orient='list'))

# Create a dictionary for each look-ahead time bin with
# corresponding heading and distance difference values

print('Creating value list')

dd_data_raw = []

for f in res_batch:
    dstd_lst = f['dst_diff']
    dst_change = f['dst_change']
    ts_lst = f['la_t']
    delta_dst_t = f['delta_dst_t']

    #     dd_data_raw.extend([d / (dc * t) for d, dc, t in
    #                         zip(dstd_lst, dst_change, ts_lst) if dc*t != 0])

    dd_data_raw.extend([distr_func(d, dct, t) for d, dct, t in
                        zip(dstd_lst, delta_dst_t, ts_lst) if dct != 0])

dd_data_p = np.percentile(dd_data_raw, 99.5)
dd_data_filt = [v for v in dd_data_raw if abs(v) < dd_data_p]

x_grid = np.linspace(min(dd_data_filt), max(dd_data_filt), 1000)
kde = gaussian_kde(dd_data_filt)  # , bw_method=0.1)
kdepdf = kde.evaluate(x_grid)

# Create final dictionary

pct_dict = {}
cdf_list = list(zip(x_grid, np.cumsum(kdepdf) / np.cumsum(kdepdf)[-1]))
for p in range(1, 100):
    pct_dict[str(p)] = [v[0] for v in cdf_list if v[1] <= p / 100][-1]

print('Saving Dictionary as pkl')
save_obj(pct_dict, 'kde_dictionary')

# res_batch = []
# bin_dict = {}
# pdf_dict = {}
#
# la_time = 1200
#
#
# # Add the heading difference and speed difference to the
# # flights returned from the database
#
# print('Preprocessing flights')
#
# for b in batch:
#
#     if len(b['ts_1']) != len(b['ts_2']):
#         print('Flights do not have same lengths')
#         continue
#
#     f_df = pd.DataFrame.from_dict(b)
#     if len(f_df) > 0:
#
#         dstc_lst = []
#         dst_lst = []
#
#         for i, r in f_df.iterrows():
#             dstc_lst.append(
#                 get_delta_dst(r['lat_1'], r['lon_1'], r['lat_2'], r['lon_2'],
#                               r['hdg_1'], r['hdg_2'], r['spd_1'], r['spd_2']))
#
#             dst_lst.append(calc_coord_dst((r['lat_1'], r['lon_1']),
#                                           (r['lat_2'], r['lon_2'])))
#
#         f_df['dst_change_i'] = dstc_lst
#         f_df['dst'] = dst_lst
#
#         f_df = f_df.dropna(how='any')
#         f_df = f_df.reset_index(drop=True)
#
#         if len(f_df) > 0:
#             f_df['dst_change'] = [f_df['dst_change_i'].iloc[0]] * len(f_df)
#             f_df['dst_diff'] = f_df['dst'] - f_df['dst'].iloc[0]
#             f_df['la_t'] = f_df['ts_1'] - f_df['ts_1'].iloc[0]
#
#             res_batch.append(f_df.to_dict(orient='list'))
#
#
# # Create a dictionary for each look-ahead time bin with
# # corresponding heading and distance difference values
#
# print('Creating value list')
#
# dd_data_raw = []
#
# for f in res_batch:
#
#     dstd_lst = f['dst_diff']
#     dst_change = f['dst_change']
#     ts_lst = f['la_t']
#
#     dd_data_raw.extend([d / (dc * t) for d, dc, t in
#                         zip(dstd_lst, dst_change, ts_lst) if dc*t != 0])
#
# dd_data_p = np.percentile(dd_data_raw, 99.5)
# dd_data_filt = [v for v in dd_data_raw if abs(v) < dd_data_p]
#
# # Create final dictionary
#
# kde = gaussian_kde(dd_data_filt, bw_method=0.1)
#
# x_grid = np.linspace(min(dd_data_filt), max(dd_data_filt), 1000)
# kdepdf = kde.evaluate(x_grid)
# pct_dict = {}
# cdf_list = list(zip(x_grid, np.cumsum(kdepdf)/np.cumsum(kdepdf)[-1]))
# for p in range(1, 100):
#     pct_dict[str(p)] = [v[0] for v in cdf_list if v[1] <= p/100][-1]
