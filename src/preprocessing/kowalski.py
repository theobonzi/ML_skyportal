import urllib
import os
import requests     
import json
import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
from astropy.utils.exceptions import AstropyWarning
import gzip
import io

from penquins import Kowalski
from dotenv import load_dotenv

load_dotenv()

def get_data_from_kowalski(all=False, nb_obj=10, dataDir = "data_kowalski/"):

    api_token = os.getenv("FRITZ_API_TOKEN")
    kowalski_token = os.getenv("KOWALSKI_API_TOKEN")

    if not api_token or not kowalski_token:
        print("Tokens not found in .env file")
        exit()

    host = "https://fritz.science/"
    headers = {'Authorization': f'token {api_token}'}

    instances = {'kowalski': {'protocol': 'https', 'port': 443, 'host': f'kowalski.caltech.edu', 'token': kowalski_token,}} 
    kowalski = Kowalski(instances=instances)
    if kowalski.ping(name="kowalski"):
        print("Connected to Kowalski")
    else:
        print("Unable to connect to Kowalski")
        exit() 

    df_bts = pd.read_csv('data/BTS.csv')
    objIds = sorted(list(set(df_bts["ZTFID"])))

    if not all:
        objIds = objIds[:nb_obj]

    print(f"Total number of objects: {len(objIds)}")

    for ii, objId in enumerate(objIds):
        print(ii, objId, float(ii / len(objIds)))

        objDirectory = os.path.join(dataDir, objId)
        if not os.path.isdir(objDirectory):
            os.makedirs(objDirectory)
        else:
            continue

        # endpoint = f"sources/{objId}/photometry"                               
        # url = urllib.parse.urljoin(host, f'/api/{endpoint}') 
        # r = requests.get(url, headers=headers) 
        # photometry = r.json()['data'] 
        # photometryFile = os.path.join(objDirectory, 'photometry.json') 
        # with open(photometryFile, 'w') as fp:  
        #     json.dump(photometry, fp)

        # endpoint = f"sources/{objId}/spectra"
        # url = urllib.parse.urljoin(host, f'/api/{endpoint}')
        # r = requests.get(url, headers=headers)
        # spectra = r.json()['data']
        # spectraFile = os.path.join(objDirectory, 'spectra.json')
        # with open(spectraFile, 'w') as fp:
        #     json.dump(spectra, fp)

        query = {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_alerts",
                "filter": {
                    # take only alerts for specified object
                    'objectId': objId,
                },
                # what quantities to recieve 
                "projection": {
                    "_id": 0,
                    "objectId": 1,

                    "candidate.candid": 1,
                    "candidate.programid": 1,
                    "candidate.fid": 1,
                    "candidate.isdiffpos": 1,
                    "candidate.ndethist": 1,
                    "candidate.ncovhist": 1,
                    "candidate.sky": 1,
                    "candidate.fwhm": 1,
                    "candidate.seeratio": 1,
                    "candidate.mindtoedge": 1,
                    "candidate.nneg": 1,
                    "candidate.nbad": 1,
                    "candidate.scorr": 1,
                    "candidate.dsnrms": 1,
                    "candidate.ssnrms": 1,
                    "candidate.exptime": 1,

                    "candidate.field": 1,
                    "candidate.jd": 1,
                    "candidate.ra": 1,
                    "candidate.dec": 1,

                    "candidate.magpsf": 1,
                    "candidate.sigmapsf": 1,
                    "candidate.diffmaglim": 1,
                    "candidate.magap": 1,
                    "candidate.sigmagap": 1,
                    "candidate.magapbig": 1,
                    "candidate.sigmagapbig": 1,
                    "candidate.magdiff": 1,
                    "candidate.magzpsci": 1,
                    "candidate.magzpsciunc": 1,
                    "candidate.magzpscirms": 1,

                    "candidate.distnr": 1,
                    "candidate.magnr": 1,
                    "candidate.sigmanr": 1,
                    "candidate.chinr": 1,
                    "candidate.sharpnr": 1,

                    "candidate.neargaia": 1,
                    "candidate.neargaiabright": 1,
                    "candidate.maggaia": 1,
                    "candidate.maggaiabright": 1,

                    "candidate.drb": 1,
                    "candidate.classtar": 1,
                    "candidate.sgscore1": 1,
                    "candidate.distpsnr1": 1,
                    "candidate.sgscore2": 1,
                    "candidate.distpsnr2": 1,
                    "candidate.sgscore3": 1,
                    "candidate.distpsnr3": 1,

                    "candidate.jdstarthist": 1,
                    "candidate.jdstartref": 1,

                    "candidate.sgmag1": 1,
                    "candidate.srmag1": 1,
                    "candidate.simag1": 1,
                    "candidate.szmag1": 1,

                    "candidate.sgmag2": 1,
                    "candidate.srmag2": 1,
                    "candidate.simag2": 1,
                    "candidate.szmag2": 1,

                    "candidate.sgmag3": 1,
                    "candidate.srmag3": 1,
                    "candidate.simag3": 1,
                    "candidate.szmag3": 1,

                    "candidate.nmtchps": 1,
                    "candidate.clrcoeff": 1,
                    "candidate.clrcounc": 1,
                    "candidate.chipsf": 1,

                    "classifications.acai_h": 1,
                    "classifications.acai_v": 1,
                    "classifications.acai_o": 1,
                    "classifications.acai_n": 1,
                    "classifications.acai_b": 1,

                    "cutoutScience": 1,
                    "cutoutTemplate": 1,
                    "cutoutDifference": 1,
                }
            }
        }

        r = kowalski.query(query)
        object_alerts = r["kowalski"]['data']
        alertsFile = os.path.join(objDirectory, 'alerts.npy') 
        np.save(alertsFile, object_alerts)

def load_kowalski_data(objId, path):

    objDirectory = os.path.join(path, objId)

    photo_df = pd.read_json(os.path.join(objDirectory, 'photometry.json'))

    if photo_df.empty:
        return None, None

    photo_df = photo_df[['obj_id', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
    photo_df['type'] = "?"
    photo_df['jd'] = photo_df['mjd'] + 2400000.5
    photo_df = photo_df.dropna(subset=['mag', 'magerr'])
    photo_df = photo_df.drop_duplicates()
    photo_df = photo_df.reset_index(drop=True)

    alertsFile = os.path.join(objDirectory, 'alerts.npy')
    object_alerts = np.load(alertsFile, allow_pickle=True)

    return photo_df, object_alerts

def decompress_fits(data):
    compressed_data = gzip.decompress(data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        with fits.open(io.BytesIO(compressed_data), ignore_missing_end=True) as hdul:
            return hdul[0].data

def process_alert(alert):
    # metadata
    metadata = alert['candidate']
    metadata_df = pd.DataFrame([metadata])
    metadata_df['obj_id'] = alert['objectId']
    columns_metadata = [
        "obj_id",
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
    metadata_df = metadata_df[columns_metadata]

    # images
    cutout_science = decompress_fits(alert['cutoutScience']['stampData'])
    cutout_template = decompress_fits(alert['cutoutTemplate']['stampData'])
    cutout_difference = decompress_fits(alert['cutoutDifference']['stampData'])

    assembled_image = np.stack((cutout_science, cutout_template, cutout_difference), axis=-1)

    return metadata_df, assembled_image

def get_first_valid_index(photometry, object_alertes):
    for i, alert in enumerate(object_alertes):
        jd_current = alert['candidate']['jd']
        photometry_filtered = photometry[photometry['jd'] < jd_current]

        filters_to_check = ['ztfr', 'ztfg', 'ztfi']
        for filt in filters_to_check:
            if (photometry_filtered['filter'] == filt).sum() >= 3:
                return i
    
    return -1

def cut_photometry(photometry, object_alertes, index):
    jd_current = object_alertes[index]['candidate']['jd']
    photometry_filtered = photometry[photometry['jd'] < jd_current]
    return photometry_filtered

# def get_data(photometry, object_alerts, index=0):

#     first_index = get_first_valid_index(photometry, object_alerts)
#     if first_index == -1:
#         return None, None
    
#     if index < first_index:
#         index = first_index

#     photometry_filtered = cut_photometry(photometry, object_alerts, index)
#     alert = object_alerts[index]

#     metadata_df, assembled_image = process_alert(alert)
#     return photometry_filtered, metadata_df, assembled_image, index

def get_data(photometry, object_alerts, index=0):
    try:
        first_index = get_first_valid_index(photometry, object_alerts)
    except Exception as e:
        print(f"Error in get_first_valid_index: {e}")
        return None, None, None, None

    if first_index == -1:
        print("No valid index found in photometry for the given object_alerts")
        return None, None, None, None
    
    if index < first_index:
        index = first_index

    try:
        photometry_filtered = cut_photometry(photometry, object_alerts, index)
    except Exception as e:
        print(f"Error in cut_photometry: {e}")
        return None, None, None, None
    
    try:
        alert = object_alerts[index]
    except Exception as e:
        print(f"Error accessing object_alerts at index {index}: {e}")
        return None, None, None, None

    try:
        metadata_df, assembled_image = process_alert(alert)
    except Exception as e:
        print(f"Error in process_alert: {e}")
        return None, None, None, None

    return photometry_filtered, metadata_df, assembled_image, index
