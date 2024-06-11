import pandas as pd
import os
import json
import tqdm
import numpy as np
import random

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

def load_all_data(photo_path, metadata_path, images_path):
    photo_df = pd.read_csv(photo_path)
    cand = pd.read_csv(metadata_path)
    triplets = np.load(images_path, mmap_mode='r')
    
    columns_metadata = [
        "objectId",
        "sgscore1", "sgscore2", 
        "distpsnr1", "distpsnr2", 
        "fwhm", 
        "magpsf", 
        "sigmapsf", 
        "ra", 
        "dec", 
        "diffmaglim", 
        "ndethist", 
        "nmtchps", 
        "ncovhist", 
        "sharpnr", 
        "scorr", 
        "sky"
    ]
    cand = cand[columns_metadata]

    cand_obj_ids = set(cand['objectId'].unique())
    photo_obj_ids = set(photo_df['obj_id'].unique())
    objIds = list(cand_obj_ids.intersection(photo_obj_ids))
    return photo_df, cand, triplets, objIds

def get_data(objId, photo_df, cand, triplets):
    one_cand = cand[cand['objectId'] == objId]
    one_image = triplets[one_cand.index[0]]
    one_photo = photo_df[photo_df['obj_id'] == objId]
    one_photo = one_photo.dropna()
    return one_photo, one_cand, one_image

def get_objId(objIds):
    return objIds[random.randint(0, len(objIds))]