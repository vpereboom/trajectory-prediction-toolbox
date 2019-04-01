import datetime
import multiprocessing
from psycopg2.extras import RealDictCursor

from tools.db_connector import get_pg_conn
from conf.config import cpu_count
from tools.flight_conflict_processor import classify_flight_pairs

conn = get_pg_conn()
cur_ts = conn.cursor(cursor_factory=RealDictCursor)
cur_ts.execute("SELECT start_ep, flight_id \
                 FROM public.adsb_flights;")

batch = cur_ts.fetchall()
x = [b['start_ep'] for b in batch]

daylst = list(set([datetime.datetime.fromtimestamp(xx).strftime('%Y-%m-%d %H')
                   for xx in x]))

eplist = [(datetime.datetime.strptime(ts, '%Y-%m-%d %H') -
           datetime.datetime(1970, 1, 1)).total_seconds() for ts in daylst]

print(len(eplist))

pool = multiprocessing.Pool(cpu_count)
res = pool.map(classify_flight_pairs, eplist)

pool.close()
pool.join()

conn.close()
cur_ts.close()
