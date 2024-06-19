from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import src.preprocessing.plot_data as plot_data
import numpy as np
import tensorflow as tf

def identify_useless_columns(group):
    mask = (group[['flux_ztfg', 'flux_ztfi', 'flux_ztfr']] != 0).any().values
    return mask

def predict_classes(photo_ready, cand_ready, image_ready):
    # photometry
    grouped = photo_ready.groupby('obj_id')
    sequences = []

    for _, group in grouped:
        mask = identify_useless_columns(group)
        # Appliquer le masque sur les colonnes de flux
        filtered_group = group[['mjd'] + list(group.columns[mask + 1])].values
        sequences.append(filtered_group)    
        
    padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

    padded_sequences = np.array(padded_sequences)

    # metadata
    metadata = cand_ready.drop(columns='obj_id')

    image_ready = np.array(image_ready)

    final_data = [padded_sequences, metadata, image_ready]

    return final_data

def predict_T2(final_data, model):
    y_pred = model.predict(final_data)

    return y_pred, np.argmax(y_pred, axis=1)

def taxonomy_prediction(final_data, model_step1, model_step2, model_step3a, model_step3b, treshold=0.6):
    # Step 1
    y_pred, y_pred_classes = predict_T2(final_data, model_step1)
    y_pred_sum = np.sum(y_pred, axis=0)

    if np.max(y_pred_sum) - np.min(y_pred_sum) > treshold:
        y_pred_classes = np.argmax(y_pred_sum)
    
    # Step 2
    if y_pred_classes == 1:
        y_pred = predict_T2(model_step2, final_data)
        y_pred_sum = np.sum(y_pred, axis=0)

        if np.max(y_pred_sum) - np.min(y_pred_sum) > treshold:
            y_pred_classes = np.argmax(y_pred_sum)

        # Step 3a
        if y_pred_classes == 0:
            y_pred = predict_T2(model_step3a, final_data)
            y_pred_sum = np.sum(y_pred, axis=0)

            if np.max(y_pred_sum) - np.min(y_pred_sum) > treshold:
                y_pred_classes = np.argmax(y_pred_sum)
        
        # Step 3b
        elif y_pred_classes == 1:
            y_pred = predict_T2(model_step3b, final_data)
            y_pred_sum = np.sum(y_pred, axis=0)

            if np.max(y_pred_sum) - np.min(y_pred_sum) > treshold:
                y_pred_classes = np.argmax(y_pred_sum)
        
    return y_pred, y_pred_classes

def plot_photometry(df, relevant_ids):
    for obj_id in relevant_ids:
        one_df = df[df['obj_id'] == obj_id]
        plot_data.plot_photometry(one_df)

def display_predictions(y_pred, obj_ids, types):
    fig, axes = plt.subplots(y_pred.shape[0], 1, figsize=(10, 20))
    for i, ax in enumerate(axes):
        sns.barplot(x=types, y=y_pred[i], ax=ax)
        ax.set_title(f'Photometry {i + 1} - Obj ID: {obj_ids[i]}')
        ax.set_ylabel('Probability')
        display_value_on_bars(ax, y_pred[i])
    plt.tight_layout()
    plt.show()

def display_value_on_bars(ax, values):
    for p, value in zip(ax.patches, values):
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{value:.2f}', 
                ha="center", va='bottom')

def weighted_average_predictions(y_pred, weights):
    valid_mask = ~np.isnan(y_pred).any(axis=1)
    valid_predictions = y_pred[valid_mask]
    valid_weights = weights[valid_mask]
    if valid_predictions.size > 0:
        return np.average(valid_predictions, axis=0, weights=valid_weights)
    return None

def final_prediction_display(final_prediction, types):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=types, y=final_prediction, palette='viridis', hue=types)
    plt.title("Final Prediction by Mean")
    plt.ylabel("Mean Probability")
    plt.xlabel("Class")
    display_value_on_bars(plt.gca(), final_prediction)
    plt.show()