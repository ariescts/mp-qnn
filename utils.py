import os
import shutil
import torch
import logging.config
import pandas as pd

from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column


def set_logging(log_file='log.txt'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog():
    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            # self.results.read_csv(path)
            self.results = pd.read_csv(path)


# def save_checkpoint(state, best_train=False, best_val=False, best_test=False,
#                     path='', filename='checkpoint.pth.tar', save_all=False):
#     filename = os.path.join(path, filename)
#     torch.save(state, filename)
#     if best_train:
#         shutil.copyfile(filename, os.path.join(path, 'model_best_train.pth.tar'))
#     if best_val:
#         shutil.copyfile(filename, os.path.join(path, 'model_best_val.pth.tar'))
#     if best_test:
#         shutil.copyfile(filename, os.path.join(path, 'model_best_test.pth.tar'))
#     if save_all:
#         shutil.copyfile(filename, os.path.join(
#             path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


def save_checkpoint(state, flag_best_val=False, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if flag_best_val:
        shutil.copyfile(filename, os.path.join(path, 'checkpoint_best_val.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(path, 'checkpoint_epoch_%s.pth.tar'
                                               % state['epoch']))


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


_optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = _optimizers[setting['optimizer']](optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' % setting['optimizer'])

        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
                    logging.debug('OPTIMIZER - setting %s = %s' % (key, setting[key]))

        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config[epoch])
    else:
        for e in range(epoch + 1):
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res