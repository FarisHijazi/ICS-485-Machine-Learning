import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt                        

# code from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fig_ax=(None, None)
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
    else:
#         print('Confusion matrix, without normalization')
        pass

#     print(cm)
    fig, ax = fig_ax
    
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax



def plot_labeled_data(data, labels, title=None):
    """
    :param chunk_size: how many features to show in each plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    r = lambda: np.random.randint(0,255) # get random color

    fig = plt.figure(figsize=(50, 50))

    classes = np.unique(labels)
    
    # plotting all combinations of features together
    ax = plt.subplot(7, 6, 1, projection='3d')

    for i in range(len(classes)):
        points = np.array([data[j] for j in range(len(data)) if labels[j] == classes[i]])
        ax.scatter(*points.T, s=8, c=('#%02X%02X%02X'%(r(),r(),r())))
    ax.set_title(title)
    return fig
