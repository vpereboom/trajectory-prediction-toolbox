import numpy as np


def separate_flights(icao_df, split_window=1800):
    icao_df = icao_df.reset_index(drop=True)
    icao_df['ts_diff'] = icao_df['ts'].diff()
    icao_df['ts_diff'].iloc[0] = 0
    icao_df['flight'] = np.nan
    ilist = [0]
    ilist.extend(icao_df.loc[icao_df['ts_diff'] > split_window].index.tolist())
    cnt = 1
    if len(ilist) > 1:
        for i in range(len(ilist) - 1):
            icao_df['flight'].iloc[ilist[i]:ilist[i + 1]] = cnt
            cnt = cnt + 1
        icao_df['flight'].iloc[ilist[-1]:] = cnt
    else:
        icao_df['flight'] = cnt

    icao_df = icao_df.drop(['ts_diff'], axis=1)
    icao_df['flight_count'] = icao_df['flight'].max()
    icao_df['flight_id'] = icao_df['icao'].astype(str) + '_' + icao_df['ts'].iloc[0].astype(int).astype(str)
    return icao_df


def create_flight_dict(df):
    dct = {}
    for name, g in df.groupby('flight_id'):
        fl_data = g[['ts', 'lat', 'lon', 'alt', 'spd', 'hdg', 'roc', 'callsign']]
        fl_data = fl_data.reset_index()
        dct[name] = {
            'flight_length': len(g),
            'icao': g['icao'].unique()[0],
            'flight_number': g['flight'].unique()[0],
            'flight_data': fl_data,
        }

    return dct
