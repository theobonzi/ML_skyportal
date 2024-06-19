import numpy as np
import pandas as pd
from tqdm import tqdm

def augment_data_with_noise(df, noise_level=0.05):
    new_entries = []
    flux_columns = ['mag', 'magerr']
    percentages = [80, 50, 20]

    for obj_id in tqdm(df['obj_id'].unique()):
        obj = df[df['obj_id'] == obj_id].sort_values('mjd').reset_index(drop=True)

        for percentage in percentages:
            subset_size = int((percentage / 100.0) * len(obj))
            
            subset_top = obj.iloc[:subset_size].copy()
            if enough_points(subset_top):
                add_noise_and_append(subset_top, obj_id, percentage, 'top', new_entries, flux_columns, noise_level)
            
            subset_end = obj.iloc[-subset_size:].copy()
            if enough_points(subset_end):
                add_noise_and_append(subset_end, obj_id, percentage, 'end', new_entries, flux_columns, noise_level)

    augmented_df = pd.concat([df] + new_entries, ignore_index=True)
    return augmented_df

# compte le nombre de points pour les filtres ztfg, ztfr, ztfi, renvoie True si au minimum 3 points pour un des filtres
def enough_points(subset):
    if len(subset[subset['filter'] == 'ztfg']) >= 3:
        return True
    if len(subset[subset['filter'] == 'ztfr']) >= 3:
        return True
    if len(subset[subset['filter'] == 'ztfi']) >= 3:
        return True
    return False

def add_noise_and_append(subset, obj_id, percentage, prefix, entries_list, flux_columns, noise_level):
    key = f"{obj_id}_{prefix}_{percentage}"
    # for col in flux_columns:
    #     noise = np.random.normal(0, noise_level * np.std(subset[col]), size=len(subset))
    #     subset[col] += noise
    subset['obj_id'] = key
    entries_list.append(subset)