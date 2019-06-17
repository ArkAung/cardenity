from fastai.vision import *


def filter_by_video_number(video_name, video_numbers):
    """Filter videos which contains the following video_number. This ensures that there are no overlap videos
    between training and valiation."""
    for num in video_numbers:
        if num in video_name:
            return True
    return False


def drop_by_video_names(video_name, video_numbers):
    """Filter videos which DOES NOT contain the following video_name.
    Those which are excluded will be used for test set"""
    for num in video_numbers:
        if num in video_name:
            return False
    return True


def databunch_from_csv(csv_name, path, img_size, vertical_flip=False, batch_size=100,
                       img_ext=None, train_folder=None, drop_list=None, val_list=None):
    """Creates a databunch from csv label file, optionally taking in validation set and/or set of files to be dropped"""
    octant_tfms = get_transforms(do_flip=True, flip_vert=vertical_flip, max_rotate=5.0,
                                 max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.2,
                                 p_lighting=0.8)
    img_item_list = ImageList.from_csv(path=path,
                                       csv_name=csv_name,
                                       folder=train_folder,
                                       suffix=img_ext)

    if drop_list is not None:
        img_item_list = (img_item_list
                         .filter_by_func(partial(drop_by_video_names, video_numbers=drop_list)))

    if val_list is not None:
        data_src = (img_item_list
                    .split_by_valid_func(partial(filter_by_video_number, video_numbers=val_list))
                    .label_from_df(label_delim=' '))
    else:
        data_src = (img_item_list
                    .split_by_rand_pct(valid_pct=0.15, seed=1337)
                    .label_from_df(label_delim=' '))

    databunch = (data_src
                 .transform(octant_tfms, size=img_size)
                 .databunch(bs=batch_size, num_workers=8)
                 .normalize(imagenet_stats))

    return databunch
