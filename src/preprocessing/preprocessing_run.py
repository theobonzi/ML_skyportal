import src.preprocessing.plot_data as plot_data
import src.preprocessing.data_augmentation as data_augmentation
import src.preprocessing.tools as tools
import src.preprocessing.gaussian_process as gp
import pandas as pd
import numpy as np

def preprocess_cand_aug(one_cand, new_ids):
    columns_metadata_nb = one_cand.columns.difference(['objectId'])
    original_id = one_cand['objectId'].iloc[0]

    cand_ready = one_cand.copy()

    for new_id in new_ids:    
        row_to_copy = one_cand[one_cand['objectId'] == original_id].iloc[0].copy()
        # Add noise to metadata
        for col in columns_metadata_nb:
            row_to_copy[col] += 0.05 * np.random.randn() * row_to_copy[col]
        row_to_copy['objectId'] = new_id
        cand_ready = pd.concat([cand_ready, pd.DataFrame([row_to_copy])], ignore_index=True)

    cand_ready = tools.robust_scale(cand_ready, columns_metadata_nb)

    return cand_ready

def prepare_gp_input(df):
    df = tools.Mag2Flux(df)
    df = tools.Normalize_mjd(df)
    return df

def apply_gaussian_process(df, kernel=None):
    return gp.process_gaussian(df, kernel=kernel, name='all')

def scale_gp_output(gp_df):
    scale_columns = [col for col in gp_df.columns if 'flux' in col]
    return tools.robust_scale(gp_df, scale_columns)

def load_data(source):
    if isinstance(source, str):
        return pd.read_csv(source)
    return source

def preprocess_data(df):
    df = df.copy()
    df['filter'] = df['filter'].replace({'sdssi': 'ztfi', 'sdssr': 'ztfr', 'sdssg': 'ztfg'})
    return df[df['filter'].isin(['ztfg', 'ztfr', 'ztfi'])]

def reset_data_indices(df):
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df['type'] = '?'
    return df

def augment_and_clean_data(df):
    df_aug = data_augmentation.augment_data_with_noise(df)
    df_aug['mag'] = df_aug['mag'].apply(lambda x: max(x, 0))
    df_aug['magerr'] = df_aug['magerr'].apply(lambda x: max(x, 0))
    return df_aug

def preprocess_photo_aug(one_photo, verbose=False):
    df = preprocess_data(one_photo)
    df = reset_data_indices(df)
    df = augment_and_clean_data(df)

    df_gp_ready = prepare_gp_input(df)

    columns = ['obj_id', 'mjd', 'flux_ztfg', 'flux_ztfi', 'flux_ztfr', 
           'flux_error_ztfg', 'flux_error_ztfi', 'flux_error_ztfr']

    final_gp = pd.DataFrame(columns=columns)
    for i in df_gp_ready['obj_id'].unique():
        one_gp = df_gp_ready[df_gp_ready['obj_id'] == i]
        gp_df = apply_gaussian_process(one_gp)
        gp_df = scale_gp_output(gp_df)
        final_gp = pd.concat([final_gp, gp_df], ignore_index=True)


    final_gp.fillna(0., inplace=True)

    if verbose:
        plot_data.plot_gp_separated_by_obj_id(final_gp)
    
    return final_gp

def rotate(image, angle):
        return np.rot90(image, k=int(angle / 90))

def flip_and_rotate(image, angle):
    image_rotated = rotate(image, angle)
    return np.flipud(image_rotated)

def normalize_images(images):
    min_val = images.min()
    max_val = images.max()
    normalized_images = 2 * (images - min_val) / (max_val - min_val) - 1
    return normalized_images

def preprocess_image_aug(one_image, new_ids, verbose=False):

    image_ready = [one_image]

    nb_length = len(new_ids)

    transformations = [
        lambda x: rotate(x, 90),
        lambda x: rotate(x, 180),
        lambda x: rotate(x, 270),
        lambda x: flip_and_rotate(x, 90),
        lambda x: flip_and_rotate(x, 180),
        lambda x: flip_and_rotate(x, 270)
    ]
    
    # Apply transformations
    for i in range(nb_length):
        transform = transformations[i % len(transformations)]
        transformed_image = transform(one_image)
        image_ready.append(transformed_image)

    # Normalize images
    image_ready = normalize_images(np.array(image_ready))

    if verbose:
        for i, image in enumerate(image_ready):
            plot_data.plot_image(image)

    return image_ready

def process_data_aug(one_photo, one_cand, one_image, verbose=False):
    photo_ready = preprocess_photo_aug(one_photo, verbose)
    new_ids = photo_ready[photo_ready['obj_id'].str.contains('_')]['obj_id'].unique()
    cand_ready = preprocess_cand_aug(one_cand, new_ids)
    image_ready = preprocess_image_aug(one_image, new_ids, verbose)
    
    return photo_ready, cand_ready, image_ready

def preprocess_photo(one_photo, verbose=False):
    df = preprocess_data(one_photo)
    df = reset_data_indices(df)

    df_gp_ready = prepare_gp_input(df)
    
    gp_df = apply_gaussian_process(df_gp_ready)
    final_gp = scale_gp_output(gp_df)
    
    final_gp['flux_ztfg'] = final_gp['flux_ztfg'].fillna(0.0)
    # final_gp['flux_ztfi'] = final_gp['flux_ztfi'].fillna(0.0)
    final_gp['flux_ztfr'] = final_gp['flux_ztfr'].fillna(0.0)
    final_gp['flux_error_ztfg'] = final_gp['flux_error_ztfg'].fillna(0.0)
    # final_gp['flux_error_ztfi'] = final_gp['flux_error_ztfi'].fillna(0.0)
    final_gp['flux_error_ztfr'] = final_gp['flux_error_ztfr'].fillna(0.0)

    if verbose:
        plot_data.plot_gp(final_gp)
    
    return final_gp

def preprocess_image(one_image, verbose=False):
    image_ready = normalize_images(np.array(one_image))

    if verbose:
        plot_data.plot_image(image_ready)
    return image_ready

def process_data(one_photo, one_cand, one_image, verbose=False):
    photo_ready = preprocess_photo(one_photo, verbose)
    image_ready = preprocess_image(one_image, verbose)

    image_reshaped = np.expand_dims(image_ready, axis=0)

    return photo_ready, one_cand, image_reshaped