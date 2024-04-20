import numpy as np
import pandas as pd
import tqdm
import math

def augment_data_with_noise(df, noise_level=0.05):
    new_entries = []
    flux_columns = ['mag', 'magerr', 'mjd']

    for obj_id in tqdm.tqdm(df['obj_id'].unique()):
        obj = df[df['obj_id'] == obj_id].sort_values('mjd').reset_index(drop=True)
        percentages = [80, 50, 20]

        for percentage in percentages:
            key = f"{obj_id}_{percentage}"
            subset_size = int((percentage / 100.0) * len(obj))
            subset = obj.iloc[:subset_size].copy()
            
            for col in flux_columns:
                noise = np.random.normal(0, noise_level * np.std(subset[col]), size=subset_size)
                subset[col] += noise
            
            subset['obj_id'] = key
            new_entries.append(subset)

    augmented_df = pd.concat([df] + new_entries, ignore_index=True)
    return augmented_df

from scipy.interpolate import CubicSpline

def magnitude_warp(df, num_knots=4, sigma=0.3):
    columns = [col for col in df.columns if 'flux' in col]
    for col in columns:
        series = df[col]
        n = len(series)
        time_points = np.linspace(0, n-1, num=n)
        warp_points = np.linspace(0, n-1, num=num_knots)
        warp_values = np.random.normal(loc=1.0, scale=sigma, size=num_knots)
        spline = CubicSpline(warp_points, warp_values)
        warp_factor = spline(time_points)
        df[col] = series * warp_factor
    return df

def shuffle_flux_features(df):
    filters = ['ztfg', 'ztfr', 'ztfi']
    flux_cols = [f'flux_{filter}' for filter in filters]
    error_cols = [f'flux_error_{filter}' for filter in filters]

    shuffled_indices = np.random.permutation(len(filters))
    
    # Reorder based on shuffled indices
    new_flux_cols = [flux_cols[i] for i in shuffled_indices]
    new_error_cols = [error_cols[i] for i in shuffled_indices]
    
    shuffled_df = df.copy()
    
    # Reassign shuffled columns
    for i in range(len(filters)):
        shuffled_df[flux_cols[i]] = df[new_flux_cols[i]].values
        shuffled_df[error_cols[i]] = df[new_error_cols[i]].values

    return shuffled_df

def add_gaussian_noise(df, variance=0.02):
    columns = [col for col in df.columns if 'flux' in col]
    for col in columns:
        noise = np.random.normal(0, np.sqrt(variance), df[col].shape)
        df[col] += noise
    return df

def augment_single_object(obj_df, augmentation_func, func_name):
    """
    Applies an augmentation function to a DataFrame containing a single time series object and modifies obj_id to include the function name.
    
    Parameters:
    - obj_df (DataFrame): DataFrame containing the data for a single object.
    - augmentation_func (callable): Function that applies the transformation.
    - func_name (str): Name of the augmentation function, used to modify obj_id.
    
    Returns:
    - DataFrame: The augmented DataFrame for the single object with updated obj_id.
    """
    augmented_obj = obj_df.copy()
    augmented_obj = augmentation_func(augmented_obj)  # Apply the augmentation directly to the DataFrame
    augmented_obj['obj_id'] = augmented_obj['obj_id'].apply(lambda x: x + '_' + func_name)
    return augmented_obj

augmentation_funcs = {
    'shuffle_features': shuffle_flux_features,
    'magnitude_warp': magnitude_warp,
    'gaussian_noise': add_gaussian_noise
}

def balanced_augmentation(data, augmentation_funcs=augmentation_funcs):
    """
    Perform data augmentation to balance the number of object IDs across different classes based on a target multiplier.
    Each class will receive a calculated number of augmentations to reach approximately the same number of object IDs.

    Parameters:
    - data (DataFrame): The full dataset.
    - augmentation_funcs (dict): Dictionary of augmentation functions.

    Returns:
    - DataFrame: The dataset augmented to balance class sizes.
    """
    # Count the number of object IDs in each class and calculate required augmentations
    class_counts = data.groupby('type')['obj_id'].nunique()
    min_class_count = class_counts.min()
    target_obj_ids = min_class_count * (len(augmentation_funcs) + 1)

    # Calculate the number of augmentations needed for each class to approach the target
    augmentations_needed = {
        class_type: math.ceil(target_obj_ids / count) - 1
        for class_type, count in class_counts.items()
    }
    # Dictionary to hold augmented data
    augmented_data_list = []

    # Apply calculated number of augmentations to each class
    for class_type, group in data.groupby('type'):
        obj_ids = group['obj_id'].unique()
        num_augmentations = augmentations_needed[class_type]
        for obj_id in tqdm.tqdm(obj_ids, desc=f"Augmenting {class_type} {num_augmentations} times"):
            obj_data = data[data['obj_id'] == obj_id]
            for func_index, (func_name, func) in enumerate(sorted(augmentation_funcs.items(), key=lambda x: x[0])):
                if func_index < num_augmentations:
                    augmented_obj = augment_single_object(obj_data, func, func_name)
                    augmented_data_list.append(augmented_obj)
                else:
                    break

    # Concatenate all augmented objects and add them to the original dataset
    augmented_data = pd.concat(augmented_data_list, ignore_index=True)
    final_data = pd.concat([data, augmented_data], ignore_index=True)

    final_data.drop(columns='index', inplace=True)
    return final_data