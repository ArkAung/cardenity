from fastai.vision import *


def databunch_from_csv(csv_name, path, img_size, vertical_flip=False, batch_size=100,
                       img_ext=None, train_folder=None, frac_validation=0.20):
    """Creates a databunch from csv label file"""
    data_transforms = get_transforms(do_flip=True, flip_vert=vertical_flip, max_rotate=5.0,
                                     max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.2,
                                     p_lighting=0.8)
    img_item_list = ImageList.from_csv(path=path,
                                       csv_name=csv_name,
                                       folder=train_folder,
                                       suffix=img_ext)

    data_src = (img_item_list
                .split_by_rand_pct(valid_pct=frac_validation, seed=1337)
                .label_from_df(label_delim=' '))

    databunch = (data_src
                 .transform(data_transforms, size=img_size)
                 .databunch(bs=batch_size, num_workers=8)
                 .normalize(imagenet_stats))

    return databunch
