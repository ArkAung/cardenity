from fastai.vision import *
from fastai.callbacks.tracker import *


class Model:
    def __init__(self, threshold, databunch, arch=None, load_model_name=None):
        self.threshold = threshold
        self.databunch = databunch
        if arch is not None:
            self.arch = arch
        else:
            self.arch = models.resnet50
        self.learn = self.get_multi_category_model(load_model_name)

    def get_multi_category_model(self, load_model_name):
        acc_02 = partial(accuracy_thresh, thresh=self.threshold)
        f_score = partial(fbeta, thresh=self.threshold)
        loaded_model = cnn_learner(self.databunch, self.arch,
                                   callback_fns=[partial(EarlyStoppingCallback, monitor='val_loss',
                                                         min_delta=0.01, patience=4)],
                                   metrics=[acc_02, f_score])
        if load_model_name is None:
            return loaded_model
        else:
            return loaded_model.load(load_model_name)

    def train_model(self, lr, high_lr=None, epochs=5, freeze=True):
        if freeze:
            self.learn.freeze()
        else:
            self.learn.unfreeze()

        if high_lr is not None:
            self.learn.fit_one_cycle(epochs, slice(lr, high_lr / 10))
        else:
            self.learn.fit_one_cycle(epochs, slice(lr))

    def validate_model(self, dataset='valid'):
        if dataset == 'valid':
            return self.learn.validate()
        elif dataset == 'train':
            return self.learn.validate(dl=self.learn.data.train_dl)

    def get_predictions(self, dataset='valid'):
        if dataset == 'valid':
            preds, y, losses = self.learn.get_preds(ds_type=DatasetType.Valid, with_loss=True)
        elif dataset == 'train':
            preds, y, losses = self.learn.get_preds(ds_type=DatasetType.Fix, with_loss=True)
        return preds, y, losses
