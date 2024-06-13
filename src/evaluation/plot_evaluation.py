import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from numpy import interp
from itertools import cycle
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def plot_multi_class_roc(y_test, y_pred, label_names):
    """
    Plot Receiver Operating Characteristic (ROC) curves for multiple classes.

    Parameters:
    - y_test (array): True binary labels in binary indicator format.
    - y_pred (array): Target scores, can either be probability estimates of the positive class,
                      confidence values, or binary decisions.
    - label_names (list): List of string names corresponding to each class.

    Outputs:
    - A plot of the ROC curves for each class, including micro-average and macro-average curves.
    """
    fpr, tpr, roc_auc = dict(), dict(), dict()
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    # Compute macro-average ROC curve and ROC area
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {:.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {:.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {} (area = {:.2f})'.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-class')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred_classes, label_names):
    """
    Plot a confusion matrix for the classification results.
    
    Parameters:
    - y_true (array): True class labels.
    - y_pred_classes (array): Predicted class indices.
    - label_names (list of str): Names corresponding to the class indices.
    
    Outputs:
    - A heatmap of the confusion matrix annotated with the count of instances in each category.
    """
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Create a DataFrame from the confusion matrix for better label handling in the heatmap
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    
    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def plot_class_accuracy(y_true, y_pred_classes, label_names):
    """
    Calculate and plot the classification accuracy for each class.
    
    Parameters:
    - y_true (array): True class labels as indices.
    - y_pred_classes (array): Predicted class indices.
    - label_names (list of str): Names corresponding to the class indices.
    
    Outputs:
    - A bar chart showing the accuracy for each class.
    """
    n_classes = len(label_names)
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)

    # Count correct predictions and total instances per class
    for i in range(len(y_true)):
        total[y_true[i]] += 1
        if y_true[i] == y_pred_classes[i]:
            correct[y_true[i]] += 1

    # Calculate accuracy as a percentage for each class
    percentages = correct / total

    # Plotting the accuracies
    plt.figure(figsize=(10, 5))
    plt.bar(label_names, percentages, color='skyblue')
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage, f'{percentage:.2%}', ha='center', va='bottom')  # Display percentage format

    plt.title('Accuracy per Class')
    plt.ylabel('Accuracy')
    plt.xlabel('Classes')
    plt.ylim(0, 1)  # Ensure the y-axis starts at 0 and ends at 1 for percentage display
    plt.show()

def resample_series(X, num_points, new_length=100):
    """
    Resamples time series data to a fixed number of points using linear interpolation.
    
    Parameters:
        X (array): The input array containing time series data of shape (n_samples, n_timesteps, n_features).
        num_points (int): The number of time points to use from the original series for interpolation.
        new_length (int): The new number of time points desired for each series.
    
    Returns:
        numpy.array: The resampled time series data.
    """
    X_resampled = np.zeros((X.shape[0], new_length, X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            interp_func = interp1d(np.arange(num_points), X[i, :num_points, j], kind='linear', fill_value='extrapolate')
            X_resampled[i, :, j] = interp_func(np.linspace(0, num_points - 1, new_length))
    return X_resampled

def early_classification_tradeoff(model, X_test, y_test, reduction_step=0.05, mcat=0.2):
    """
    Evaluates the trade-off between classification accuracy and average days for early prediction.

    Parameters:
        model (Model): The trained classification model.
        X_test (array): Test dataset.
        y_test (array): True labels for the test dataset.
        reduction_step (float): Fractional reduction of the series length in each iteration.
        mcat (float): Minimum classification accuracy threshold to continue reduction.

    Returns:
        tuple: Lists containing actual average days used and their corresponding accuracies.
    """
    accuracies = []
    average_days = []
    current_index = X_test.shape[1]  # Start with the maximum length of series

    while current_index > 0 and (len(accuracies) == 0 or accuracies[-1] >= mcat):
        # Calculate average days from the first time point to the current index
        avg_days = np.mean(X_test[:, :current_index, 0], axis=1) - X_test[:, 0, 0]
        mean_avg_days = np.mean(avg_days)

        truncated_X_test = X_test[:, :current_index, :]  # Exclude mjd from features for prediction
        resampled_X_test = resample_series(truncated_X_test, current_index)
        y_pred_proba = model.predict(resampled_X_test)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)

        accuracies.append(accuracy)
        average_days.append(mean_avg_days)

        # Reduce the number of points by the reduction step
        current_index = int(current_index * (1 - reduction_step))
        if current_index <= 1:  # Prevent going to zero or negative indices
            break

    plt.figure(figsize=(10, 5))
    plt.plot(average_days, accuracies, marker='o', linestyle='-')
    plt.title('Tradeoff between Accuracy and Average Days to Classification')
    plt.xlabel('Average Days to Classification')
    plt.ylabel('Classification Accuracy')
    plt.grid(True)
    plt.show()

    return average_days, accuracies

def plot_supernova_classification(df, types):
    plt.figure(figsize=(12, 6))

    plt.fill_between(df['alertes'], 0, df['type_1'], color='orange', alpha=0.7, label=types[0])

    plt.fill_between(df['alertes'], df['type_1'], 1, color='blue', alpha=0.7, label=types[1])

    # Labels and title
    plt.xlabel('Alertes')
    plt.ylabel('Probability')
    plt.title('Supernova Classification Over Time')
    plt.ylim(0, 1)

    plt.legend()

    plt.show()

# def plot_interactive_supernova_classification(df, types):
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=df['alertes'], 
#         y=df['type_1'],
#         fill='tozeroy',
#         mode='none',
#         name=types[0],
#         hoverinfo='text',
#         text=df['type_1'],
#         fillcolor='orange'
#     ))

#     fig.add_trace(go.Scatter(
#         x=df['alertes'], 
#         y=df['type_2'] + df['type_1'],
#         fill='tonexty',
#         mode='none',
#         name=types[1],
#         hoverinfo='text',
#         text=df['type_2'],
#         fillcolor='blue'
#     ))

#     fig.update_layout(
#         title='Supernova Classification Over Time',
#         xaxis_title='Alertes',
#         yaxis_title='Probability',
#         yaxis=dict(range=[0, 1]),
#         hovermode='x unified'
#     )

#     fig.show()

def plot_interactive_supernova_classification(df, types, counts_dict):
    fig = go.Figure()

    hover_headers = [
        f"Alerte: {a}<br>Prediction: {types[p]}<br>---<br>ztfr: {counts_dict.get(a, {}).get('ztfr', 0)} | ztfg: {counts_dict.get(a, {}).get('ztfg', 0)} | ztfi: {counts_dict.get(a, {}).get('ztfi', 0)}<br>---"
        for a, p in zip(df['alertes'], df['prediction'])
    ]

    hover_text_1 = [
        f"{types[0]}: {round(df.loc[i, 'type_1']*100, 4)}%"
        for i in range(len(df))
    ]

    hover_text_2 = [
        f"{types[1]}: {round(df.loc[i, 'type_2']*100, 4)}%"
        for i in range(len(df))
    ]

    fig.add_trace(go.Scatter(
        x=df['alertes'], 
        y=df['type_1'],
        fill='tozeroy',
        mode='none',
        name=types[0],
        hoverinfo='text',
        text=[f"<br>{text}" for header, text in zip(hover_headers, hover_text_1)],
        fillcolor='orange'
    ))

    fig.add_trace(go.Scatter(
        x=df['alertes'], 
        y=df['type_2'] + df['type_1'],
        fill='tonexty',
        mode='none',
        name=types[1],
        hoverinfo='text',
        text=[f"{header}<br>{text}" for header, text in zip(hover_headers, hover_text_2)],
        fillcolor='blue'
    ))

    fig.update_layout(
        title='Supernova Classification Over Time',
        xaxis_title='Alertes',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )

    fig.show()