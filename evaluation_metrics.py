import torch


def true_pos(y_true, y_pred, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim)


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


def f1_score(gts: torch.Tensor, preds: torch.Tensor, dim=(-1, -2)):
    gts = gts.float()
    preds = preds.float()

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1
