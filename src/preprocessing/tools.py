import numpy as np
from sklearn.preprocessing import RobustScaler

def categorize_type(event_type):
    if 'SN II' in event_type:
        return 'SN II'
    elif 'SN I' in event_type:
        return 'SN I'
    else:
        return 'Other'
    
def Mag2Flux(df):
    df_copy = df.dropna().copy()
    df_copy['flux'] = 10 ** (-0.4 * (df_copy['mag'] - 23.9))
    df_copy['flux_error'] = (df_copy['magerr'] / (2.5 / np.log(10))) * df_copy['flux']

    df_copy = df_copy[['obj_id', 'mjd', 'flux', 'flux_error', 'filter', 'type']]
    return df_copy

def Normalize_mjd(df):
    df_copy = df.dropna().copy()

    df_copy['mjd'] = df_copy.groupby('obj_id')['mjd'].transform(lambda x: x - np.min(x))

    df_copy.reset_index(drop=True, inplace=True)
    return df_copy

def count_obj_by_type(df):
    obj_id_count_per_type = df.groupby('type')['obj_id'].nunique()
    print(obj_id_count_per_type)

# def robust_scale(dataframe, scale_columns):
#     scaler = RobustScaler()
#     scaler = scaler.fit(dataframe[scale_columns].to_numpy())
#     dataframe.loc[:, scale_columns] = scaler.transform(
#         dataframe[scale_columns].to_numpy()
#     )
#     return dataframe

def robust_scale(dataframe, scale_columns):
    scaler = RobustScaler()
    scaler.fit(dataframe[scale_columns].to_numpy())
    dataframe[scale_columns] = scaler.transform(dataframe[scale_columns].to_numpy())
    
    return dataframe