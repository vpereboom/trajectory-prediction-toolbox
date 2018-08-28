import os

from data_parsers.adsb_raw_parser import parse_adsb_file

# File path to raw data files
fpath = '/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/to_parse/adsb'
fpath_parsed = '/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/parsed/adsb'
files = [f for f in os.listdir(fpath)]

for f_csv in files:
    f_in = os.path.join(fpath, f_csv)
    f_parsed = os.path.join(fpath_parsed, f_csv)
    print(f_in)
    parse_adsb_file(f_in)
    os.rename(f_in, f_parsed) #Move file from "to_parse" to "parsed" folder
