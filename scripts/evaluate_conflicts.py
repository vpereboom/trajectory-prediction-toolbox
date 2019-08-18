import multiprocessing
import psycopg2 as psql
from psycopg2.extras import RealDictCursor
import time

import sys
sys.path.append("..")
from tools.conflict_handling import *
from tools.db_connector import get_pg_conn


def f1_score(p_dict):
    tp, fp, tn, fn = p_dict['TP'][0], p_dict['FP'][0], p_dict['TN'][0], \
                     p_dict['FN'][0]

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    f1 = 2 * (1 / (1 / prec + 1 / rec))

    return f1


def fx_score(p_dict, x):
    tp, fp, tn, fn = p_dict['TP'][0], p_dict['FP'][0], p_dict['TN'][0], \
                     p_dict['FN'][0]

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    fx = (1 + x**2) * (prec * rec) / ((x**2 * prec) + rec)

    return fx


def load_obj(name):
    return pickle.load(open(name, "rb"))


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as filename:
        pickle.dump(obj, filename, pickle.HIGHEST_PROTOCOL)


type_list = ['deterministic', 'probabilistic', 'intent']
ti = 1

processed_dict = {}

n_cores = multiprocessing.cpu_count()

# Preprocessing Data

pool = multiprocessing.Pool(n_cores)

# if any(t in type_list for t in ['deterministic', 'probabilistic']):
#
#     conn = get_pg_conn()
#     cur_read = conn.cursor(cursor_factory=RealDictCursor)
#     cur_read.execute("SELECT ts_1, ts_2, lat_1, lon_1, lat_2,\
#                      lon_2, hdg_1, hdg_2, spd_1, spd_2 \
#                      FROM public.converging_ctfm_flights limit 250;")
#     batch = cur_read.fetchall()
#
#     cur_read = conn.cursor(cursor_factory=RealDictCursor)
#     cur_read.execute("SELECT ts_1, ts_2, lat_1, lon_1, lat_2,\
#                      lon_2, hdg_1,hdg_2, spd_1, spd_2 \
#                      FROM public.ctfm_conflicts limit 1300;")
#     batch_2 = cur_read.fetchall()
#     batch.extend(batch_2)
#     b_list = [batch[i:i + n_cores] for i in range(0, len(batch), n_cores)]
#     conn.close()
#
#     processed_batch = []
#
#     print('Preprocessing %s Data' % 'Intent and/or Probabilistic')
#     for i, r in enumerate(pool.imap_unordered(preprocess_det_conflicts,
#                                               b_list)):
#         processed_batch.append(r)
#         print("Progress: %d / %d" % (i+1, len(b_list)))
#
#     if 'deterministic' in type_list:
#         processed_dict['deterministic'] = processed_batch
#
#     if 'probabilistic' in type_list:
#         processed_dict['probabilistic'] = processed_batch


if 'intent' in type_list:
    conn = get_pg_conn()
    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT * FROM public.converging_ctfm_flights limit 300;")
    batch = cur_read.fetchall()

    cur_read = conn.cursor(cursor_factory=RealDictCursor)
    cur_read.execute("SELECT * FROM public.ctfm_conflicts limit 4000;")
    batch_2 = cur_read.fetchall()

    batch.extend(batch_2)
    conn.close()

    b_list = [batch[i:i + n_cores] for i in range(0, len(batch), n_cores)]

    processed_batch_intent = []
    processed_batch_reg = []

    print('Preprocessing %s Data' % 'Intent')

    for i, r in enumerate(pool.imap_unordered(preprocess_intent_conflicts,
                                              b_list)):
        processed_batch_intent.append(r[0])
        processed_batch_reg.append(r[1])
        print("Progress: %d / %d" % (i+1, len(b_list)))

    processed_dict['intent'] = processed_batch_intent
    processed_dict['probabilistic'] = processed_batch_reg
    processed_dict['deterministic'] = processed_batch_reg

pool.close()
pool.join()


for processing_type in type_list:

    processed_batch = processed_dict[processing_type]

    # Evaluating Conflicts
    print('Evaluating %d Conflicts' % len(processed_batch))
    pool2 = multiprocessing.Pool(n_cores)

    res = []

    if processing_type == 'deterministic':
        for i, r in enumerate(pool2.imap_unordered(process_flight_batch_tstep,
                              processed_batch)):
            res.append(r)
            print("Progress: %d / %d" % (i+1, len(processed_batch)))

    elif processing_type == 'probabilistic':
        for i, r in enumerate(pool2.imap_unordered(process_proba_flight_batch_tstep,
                                                   processed_batch)):
            res.append(r)
            print("Progress: %d / %d" % (i+1, len(processed_batch)))

    elif processing_type == 'intent':
        for i, r in enumerate(pool2.imap_unordered(process_intent_flight_batch_tstep,
                                                   processed_batch)):
            res.append(r)
            print("Progress: %d / %d" % (i+1, len(processed_batch)))

    else:
        processed_batch = []
        print('Processing type not valid')
        res = []

    pool2.close()
    pool2.join()

    ttc_list = []
    for r in res:
        ttc_list.extend(r)

    perf_dict = create_la_performance_dict(ttc_list)

    save_obj(perf_dict, '%s_%s' % ('perf_dict', processing_type))
    save_obj(ttc_list, '%s_%s' % ('ttc_list', processing_type))
