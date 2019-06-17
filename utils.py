import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def prepare_annotations(annotations, class_names):
    arr_annotations = []
    for anno in annotations:
        img_path = anno[0][0]
        bbox_x1 = anno[1][0][0]
        bbox_x2 = anno[2][0][0]
        bbox_y1 = anno[3][0][0]
        bbox_y2 = anno[4][0][0]
        indx_class_name = anno[5][0][0]
        train_test = anno[6][0][0]
        string_class_name = class_names[indx_class_name - 1][0]
        arr_annotations.append({'img_path': img_path,
                                'bbox_x1': bbox_x1,
                                'bbox_x2': bbox_x2,
                                'bbox_y1': bbox_y1,
                                'bbox_y2': bbox_y2,
                                'indx_class_name': indx_class_name,
                                'string_class_name': string_class_name,
                                'train_test': train_test})

    df = pd.DataFrame(arr_annotations, columns=['img_path', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2',
                                                'indx_class_name', 'string_class_name', 'train_test'])
    return df
