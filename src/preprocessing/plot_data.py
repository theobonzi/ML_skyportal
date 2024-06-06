import matplotlib.pyplot as plt
import numpy as np

def plot_types_distributions(data, form='bar'):

    if form not in ['bar', 'pie']:
        print('Invalid form, please choose between bar and pie')
        return
    
    count = data['type'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.title('Distribution of Types')
    plt.ylabel('Types')
    plt.xlabel('Count')

    if form == 'bar':
        count.plot(kind='bar')
        for i, v in enumerate(count):
            plt.text(i, v, str(v), ha='center', rotation=45, fontsize=10)
    else:
        count.plot(kind='pie', autopct='%1.1f%%', shadow=True)

    plt.show()

def plot_photometry(lc, color_dict=None):
    if color_dict is None:
        color_dict = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'y',
              'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y',
              'atlasc': 'cyan', 'atlaso': 'orange',}
    
    fig, ax1 = plt.subplots(1, 1, figsize=(9,6))
    ymin, ymax = np.inf, -np.inf

    for f in set(lc['filter']):
        tf = lc[lc['filter'] == f]
        
        tf_det = tf[tf['mag'] >= 3.]
        tf_ul = tf
        if 'snr' in tf.columns:
            tf_ul = tf[tf['snr'] < 3]

        ax1.errorbar(tf_det['mjd'].values,
                     tf_det['mag'], yerr=tf_det['magerr'],
                     color=color_dict[f], markeredgecolor='k',
                     label=f, marker='o')
        if np.min(tf_det['mag']) < ymin:
            ymin = np.min(tf_det['mag'])
        if np.max(tf_det['mag']) > ymax:
            ymax = np.max(tf_det['mag'])
                     
        if len(tf_ul) != 0:
            # ax1.errorbar(tf_ul['mjd'].values, tf_ul['limiting_mag'],
            #              markeredgecolor=color_dict[f],
            #              markerfacecolor='w', fmt='v', linestyle='None')
            # plt.plot([],[], 'kv', markeredgecolor='k', markerfacecolor='w',
            #          label='Upper limits')
            
            if np.min(tf_det['mag']) < ymin:
                ymin = np.min(tf_det['mag'])
            if np.max(tf_det['mag']) > ymax:
                ymax = np.max(tf_det['mag'])
    
    plt.gca().invert_yaxis()
    ax1.set_xlabel("MJD", fontsize=18)
    ax1.set_ylabel("Magnitude (AB)", fontsize=18)
    plt.legend()

    ax1.set_title(f"{lc['obj_id'].values[0]} - {lc['type'].values[0]}")
    plt.show()

def plot_gp(obj_model, number_col=4, show_title=True, show_legend=True):
    color_dict = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'yellow'}
    f, ax = plt.subplots(figsize=(10, 6))

    # Plotting model data if available
    if obj_model is not None:
        # Iterate over the filters present in the obj_model DataFrame
        for column in obj_model.columns:
            if 'flux' in column and column != 'mjd':
                filter_name = column.split('_')[1]
                if filter_name in color_dict:
                    if not all(obj_model[column] == -1):
                        ax.plot(obj_model['mjd'], obj_model[column], label=f'Model {filter_name}', color=color_dict[filter_name])
    
                        # Plot error band if error data is available
                        error_column = f'flux_error_{filter_name}'
                        if error_column in obj_model.columns:
                            model_flux_error = obj_model[error_column]
                            ax.fill_between(obj_model['mjd'], obj_model[column]-model_flux_error, obj_model[column]+model_flux_error, color=color_dict[filter_name], alpha=0.20)

    ax.set_xlabel('Time (mjd)')
    ax.set_ylabel('Flux')
    if show_title:
        obj_id = obj_model['obj_id'].iloc[0]
        ax.set_title(f'Light Curve for Object ID: {obj_id} | Type: {obj_model["type"].iloc[0]}')
    if show_legend:
        ax.legend(ncol=number_col)
    
    plt.show()

def plot_image(image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['Science', 'Template', 'Difference']

    for i, ax in enumerate(axes):
        ax.imshow(image[:, :, i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.show()

def plot_all(one_photo, one_cand, one_image):
    plot_photometry(one_photo)
    plot_image(one_image)
    print(one_cand.head())

def plot_light_curve(ax, data, color_dict, show_title, show_legend, number_col):
    for column in data.columns:
        if 'flux' in column and column != 'mjd':
            filter_name = column.split('_')[1]
            if filter_name in color_dict:
                if not all(data[column] == -1):
                    ax.plot(data['mjd'], data[column], label=f'Model {filter_name}', color=color_dict[filter_name])

                    error_column = f'flux_error_{filter_name}'
                    if error_column in data.columns:
                        model_flux_error = data[error_column]
                        ax.fill_between(data['mjd'], data[column] - model_flux_error, data[column] + model_flux_error, color=color_dict[filter_name], alpha=0.20)

    ax.set_xlabel('Time (mjd)')
    ax.set_ylabel('Flux')
    if show_title:
        obj_id = data['obj_id'].iloc[0]
        obj_type = data['type'].iloc[0]
        ax.set_title(f'Light Curve for Object ID: {obj_id} | Type: {obj_type}')
    if show_legend:
        ax.legend(ncol=number_col)

def plot_gp_separated_by_obj_id(df, number_col=4, show_title=True, show_legend=True):
    color_dict = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'yellow'}
    
    obj_ids = df['obj_id'].unique()
    
    obj_model = df[df['obj_id'] == obj_ids[0]]
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_light_curve(ax1, obj_model, color_dict, show_title, show_legend, number_col)
    plt.show()
    
    remaining_obj_ids = obj_ids[obj_ids != obj_ids[0]][:6]
    fig2, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, rem_obj_id in zip(axes, remaining_obj_ids):
        rem_obj_model = df[df['obj_id'] == rem_obj_id]
        plot_light_curve(ax, rem_obj_model, color_dict, show_title, show_legend, number_col)
    
    plt.tight_layout()
    plt.show()