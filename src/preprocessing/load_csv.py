import pandas as pd
import os
import json
import tqdm

def load_BTS_data(BTS_path):
    """
    Load the BTS data from the csv file.
    """
    # Load the data
    df = pd.read_csv(BTS_path)
    df.rename(columns={'ZTFID': 'objectId'}, inplace=True)
    return df


def load_all_photometry(df, dataDir=None, save=False):
    if dataDir is None:
        print('Please provide the path to the data directory')
        return
    
    res_df = pd.DataFrame()
    for obj_id in tqdm.tqdm(df['objectId']):
        try:
            objDirectory = os.path.join(dataDir, obj_id)
            photometryFile = os.path.join(objDirectory, 'photometry.json')
            with open(photometryFile) as f:
                photometry = json.load(f)

            photometry_df = pd.DataFrame(photometry)            
            photometry_df = photometry_df[['obj_id', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
            
            type_obj = df[df['objectId'] == obj_id]['type'].values[0]
            photometry_df['type'] = type_obj
            
            res_df = pd.concat([res_df, photometry_df])
        except Exception as e:
            print(f"Failed for {obj_id}: {e}")
            continue

    res_df.reset_index(drop=True, inplace=True)

    if save:
        types_str = '_'.join(df['type'].unique()) if hasattr(df['type'].unique(), '__iter__') else str(df['type'].unique())
        filename = f'photometry_{types_str}.csv'
        filename = filename.replace(' ', '_')
        res_df.to_csv(filename, index=False)
        print(f'File {filename} saved successfully')

    return res_df
