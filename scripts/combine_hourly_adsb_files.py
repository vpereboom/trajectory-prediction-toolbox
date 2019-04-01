import pandas as pd
import os


def convert_hex(hexstr):
    num_of_bits = len(hexstr) * 4
    binstr = bin(int(hexstr, 16))[2:].zfill(int(num_of_bits))
    return int(binstr[0:5], 2)


def filter_df(df_adsb):
    df_adsb['df'] = df_adsb['msg'].apply(lambda x: convert_hex(x))
    df_filt = df_adsb[df_adsb['df'] == 17 or df_adsb['df'] == 18]
    return df_filt


if __name__=='__main__':
    hour_folder_path = '/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/raw/adsb/hourly'
    day_folder_path = '/mnt/59069d64-9ea5-4e20-9f29-fe60f14628ea/Thesis_data/raw/adsb/days'
    hour_day_folders = [f for f in os.listdir(hour_folder_path) if '.csv' in f]

    for hd in hour_day_folders:
        hours_to_combine = [fh for fh in os.listdir(os.path.join(hour_folder_path, hd)) if '.csv' in fh]
        combined_df = pd.DataFrame()

        for f_csv in hours_to_combine:
            f_in = os.path.join(os.path.join(hour_folder_path, hd), f_csv)
            f_parsed = os.path.join(day_folder_path, f_csv)
            combined_df.append(filter_df(pd.DataFrame.from_csv(f_in)))

            day_name = '%s.csv' % hd

        combined_df.to_csv(os.path.join(day_folder_path, day_name))
