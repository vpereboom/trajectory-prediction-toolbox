import sys
sys.path.append("..")

from scipy.stats import gaussian_kde
from psycopg2.extras import RealDictCursor
import multiprocessing
from tools.flight_projection import *
from tools.db_connector import get_pg_conn
from tools.conflict_handling import gamma_angle, get_delta_dst
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


def create_dst_bound_dict(la_ti):
    hdg_dst_dict = {}

    dstc_data_raw = [v[1] for v in bin_dict['%d' % la_ti]]
    dst_data_raw = [v[0] for v in bin_dict['%d' % la_ti]]
    if len(dstc_data_raw) == len(dst_data_raw):

        print('Creating kde for la %d' % la_ti)

        dat = [v for v in list(zip(dstc_data_raw, dst_data_raw)) if
               all(~np.isnan(vi) for vi in v)]

        hl, dl = zip(*dat)
        dstc_data_raw, dst_data_raw = (list(hl), list(dl))
        data = np.vstack([dstc_data_raw, dst_data_raw])

        kde = gaussian_kde(data, bw_method=0.1)

        dst_space = np.linspace(min(dst_data_raw), max(dst_data_raw), gsize)

        pdf_dict['%d' % la_ti] = kde
        hdg_dst_dict['%d' % la_ti] = {}

        for dci in range(0, 181):

            try:
                hdg_dst_dict['%d' % la_ti]['%d' % dci] = \
                    sample_dst_kde(kde, dci, dst_space, 0.2, gsize)

            except Exception as e:
                print(e)
                hdg_dst_dict['%d' % la_ti]['%d' % dci] = (np.nan, np.nan)

        return hdg_dst_dict

    else:
        print('Heading and distance arrays not same length')
        return None


# Get database connection and define global variables

conn = get_pg_conn()
cur_read = conn.cursor(cursor_factory=RealDictCursor)
cur_read.execute("SELECT ts_1, ts_2, lat_1, lon_1, lat_2, lon_2, \
                    hdg_1, hdg_2, spd_1, spd_2 \
                    FROM public.overlapping_flights limit 2000;")
batch = cur_read.fetchall()

res_batch = []
bin_dict = {}
pdf_dict = {}

la_time = 1200
bin_sec = 20
gsize = 200


# Add the heading difference and speed difference to the
# flights returned from the database

print('Preprocessing flights')

for b in batch:

    #     print(b.keys())
    # filter_flag = 1
    #
    # if filter_flag == 1:
    #     ix = filter_gaps(b['ts_1'])
    #     if not ix:
    #         ix = (0, -1)
    # else:
    #     ix = (0, -1)
    #
    # for k in b.keys():
    #     if '_1' in k:
    #         b[k] = b[k][ix[0]:ix[1]]
    #     if '_2' in k:
    #         b[k] = b[k][ix[0]:ix[1]]

    if len(b['ts_1']) != len(b['ts_2']):
        print('Flights do not have same lengths')
        continue

    f_df = pd.DataFrame.from_dict(b)
    if len(f_df) > 0:
        f_df['spd_diff'] = f_df['spd_1'] - f_df['spd_2']

        dstc_lst = []
        hdg_diff_lst = []
        dst_lst = []

        for i, r in f_df.iterrows():
            dstc_lst.append(
                get_delta_dst(r['lat_1'], r['lon_1'], r['lat_2'], r['lon_2'],
                              r['hdg_1'], r['hdg_2'], r['spd_1'], r['spd_2']))

            _g, _a, _b = gamma_angle(r['lat_1'], r['lon_1'], r['lat_2'],
                                     r['lon_2'], r['hdg_1'], r['hdg_2'])
            hdg_diff_lst.append(_g)

            dst_lst.append(calc_coord_dst((r['lat_1'], r['lon_1']),
                                          (r['lat_2'], r['lon_2'])))

        f_df['dst_change_i'] = dstc_lst
        f_df['hdg_diff_i'] = hdg_diff_lst
        f_df['dst'] = dst_lst

        f_df = f_df.dropna(how='any')
        f_df = f_df.reset_index(drop=True)

        if len(f_df) > 0:
            f_df['dst_change'] = [f_df['dst_change_i'].iloc[0]] * len(f_df)
            f_df['hdg_diff'] = [f_df['hdg_diff_i'].iloc[0]] * len(f_df)
            f_df['dst_diff'] = f_df['dst'] - f_df['dst'].iloc[0]
            f_df['la_t'] = f_df['ts_1'] - f_df['ts_1'].iloc[0]

            #         bi = f_df.to_dict(type='list')
            res_batch.append(f_df.to_dict(orient='list'))


# Create a dictionary for each look-ahead time bin with
# corresponding heading and distance difference values

print('Creating dictionary')

for f in res_batch:

    dstd_lst = f['dst_diff']
    dst_change = f['dst_change']
    ts_lst = f['la_t']

    for b in range(int(la_time / bin_sec)):
        bmin = b * bin_sec
        bmax = (b + 1) * bin_sec
        if str(bmax) not in list(bin_dict.keys()):
            bin_dict[str(bmax)] = []

        bin_dict[str(bmax)].extend([(d, dc, t) for d, dc, t in
                                    zip(dstd_lst, dst_change, ts_lst)
                                    if t >= bmin and t <= bmax])


# Create a kde estimator for each look-ahead time bin

pool_cpu_size = multiprocessing.cpu_count()
pool = multiprocessing.Pool(pool_cpu_size)
res_list = pool.map(create_dst_bound_dict,
                    [int(la) for la in bin_dict.keys()])
pool.close()
pool.join()


# Create final dictionary

print('Saving Dictionary as pkl')
final_hdg_dst_dict = {}

for r in res_list:
    key = list(r.keys())[0]
    final_hdg_dst_dict[key] = r[key]

save_obj(final_hdg_dst_dict, 'kde_dictionary')
