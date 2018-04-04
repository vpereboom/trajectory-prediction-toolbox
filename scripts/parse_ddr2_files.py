import os

from data_parsers.ddr2_raw_parser import parse_ddr2_file

# File path to raw data files
fpath = '/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/to_parse/ddr2'
files = [f for f in os.listdir(fpath)]

cols = ['seg_id', 'org', 'dst', 'ac_type', 't_seg_b', 't_seg_e', 'fl_seg_b',
            'fl_seg_e', 'status', 'callsgn', 'dd_seg_b', 'dd_seg_e', 'lat_seg_b',
            'lon_seg_b', 'lat_seg_e', 'lon_seg_e', 'flight_id', 'seq', 'seg_len',
            'seg_par']

for f_csv in files:
    f_in = os.join(fpath, f_csv)
    ftype = f_csv.strip('.so6').split('_')[-1]
    print(f_in)
    print(ftype)
    parse_ddr2_file(f_in, cols, ftype)