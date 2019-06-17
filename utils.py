import PIL.Image as Image
import matplotlib.pytplot as plt
import numpy as np


def display_image(image_pth):
    """Load image and display image"""
    im = Image.open(image_pth)
    plt.imshow(np.asarray(im))
    plt.show()


def threshold_predictions(preds, classes, prediction_threshold=0.5):
    """Get labelled preds by thresholding the raw probability values and joining them up"""
    labelled_preds = [' '.join([classes[i] for i, p in enumerate(pred) if p > prediction_threshold])
                      for pred in preds]
    return labelled_preds


def check_acc(y, preds):
    """Check accuracy by checking whether parts of y and parts of predictions are equal
    The main reason is that the model makes predictions on model names and manufacturer names independently so the
    order of prediction when concatenated may be <manufacture_name> <model_name> or <model_name> <manufacture name>"""
    arr_acc = []
    for i in range(len(y)):
        arr_acc.append(set(y[i].split()) == set(preds[i].split()))
    return np.mean(arr_acc)
