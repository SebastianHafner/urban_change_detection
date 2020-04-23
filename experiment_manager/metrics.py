import torch
from sklearn.metrics import roc_auc_score, roc_curve
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


class MultiThresholdMetric():
    def __init__(self, threshold):

        # FIXME Does not operate properly

        '''
        Takes in rasterized and batched images
        :param y_true: [B, H, W]
        :param y_pred: [B, C, H, W]
        :param threshold: [Thresh]
        '''

        self._thresholds = threshold[ :, None, None, None, None] # [Tresh, B, C, H, W]
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def _normalize_dimensions(self):
        ''' Converts y_truth, y_label and threshold to [B, Thres, C, H, W]'''
        # Naively assume that all of existing shapes of tensors, we transform [B, H, W] -> [B, Thresh, C, H, W]
        self._thresholds = self._thresholds[ :, None, None, None, None] # [Tresh, B, C, H, W]
        # self._y_pred = self._y_pred[None, ...]  # [B, Thresh, C, ...]
        # self._y_true = self._y_true[None,:, None, ...] # [Thresh, B,  C, ...]

    def add_sample(self, y_true:torch.Tensor, y_pred):
        y_true = y_true.bool()[None,...] # [Thresh, B,  C, ...]
        y_pred = y_pred[None, ...]  # [Thresh, B, C, ...]
        y_pred_offset = (y_pred - self._thresholds + 0.5).round().bool()

        self.TP += (y_true & y_pred_offset).sum(dim=self._data_dims).float()
        self.TN += (~y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FP += (y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FN += (~y_true & y_pred_offset).sum(dim=self._data_dims).float()

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall

    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom

class MultiClassF1():
    def __init__(self, ignore_last_class = False):

        # FIXME Does not operate properly

        '''
        Takes in rasterized and batched images
        :param y_true: [B, H, W]
        :param y_pred: [B, C, H, W]
        :param threshold: [Thresh]
        '''

        self._data_dims = (0, 2, 3) # For a B/W image, it should be [Thresh, B, C, H, W],
        self.ignore_last_class = ignore_last_class
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def add_sample(self, y_true:torch.Tensor, y_pred):
        if self.ignore_last_class:
            # Ignore background classes

            y_pred = y_pred[:, :-1]
            y_true = y_true[:, :-1]
            y_true_mask = y_true[:, [-1]]


        y_true = y_true.bool() # [B,  C, ...]
        # Make y_pred one hot along dim C
        y_pred_argmax = torch.argmax(y_pred, dim=1, keepdim=True)  # [B, C, ...]
        y_pred = torch.zeros_like(y_pred, dtype=torch.bool).scatter_(1, y_pred_argmax,True)

        self.TP += (y_true & y_pred).sum(dim=self._data_dims).float()
        self.TN += ((~y_true & ~y_pred) * y_true_mask).sum(dim=self._data_dims).float()
        self.FP += ((~y_true & y_pred) * y_true_mask).sum(dim=self._data_dims).float()
        self.FN += (y_true & ~y_pred).sum(dim=self._data_dims).float()


    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self, include_bg=False):
        self.TP = self.TP.clamp(10e-05)
        self.TN = self.TN.clamp(10e-05)
        self.FP = self.FP.clamp(10e-05)
        self.FN = self.FN.clamp(10e-05)
        individual_f1 = 2 * self.TP / (2 * self.TP + self.FN + self.FP)
        if not include_bg:
            individual_f1 = individual_f1[..., :-1]
        f1 = len(individual_f1) / (individual_f1 ** -1).sum()

        f1 = f1.cpu().item()
        individual_f1 = individual_f1.cpu().numpy()
        return f1, individual_f1

def true_pos(y_true, y_pred, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim) # Only sum along H, W axis, assuming no C


def false_pos(y_true, y_pred, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def false_neg(y_true, y_pred, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def precision(y_true, y_pred, dim):
    denom = (true_pos(y_true, y_pred, dim) + false_pos(y_true, y_pred, dim))
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom

def recall(y_true, y_pred, dim):
    denom = (true_pos(y_true, y_pred, dim) + false_neg(y_true, y_pred, dim))
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom

def f1_score(gts:torch.Tensor, preds:torch.Tensor, multi_threashold_mode=False, dim=(-1, -2)):
    # FIXME Does not operate proper
    gts = gts.float()
    preds = preds.float()

    if multi_threashold_mode:
        gts = gts[:, None, ...] # [B, Thresh, ...]
        gts = gts.expand_as(preds)

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


def roc_score(y_true:torch.Tensor, y_preds:torch.Tensor, ):
    y_preds = y_preds.flatten().cpu().numpy()
    y_true = y_true.flatten().cpu().numpy()

    curve = roc_curve(y_true, y_preds, pos_label=1,  drop_intermediate=False)
    # print(curve)
    return curve
