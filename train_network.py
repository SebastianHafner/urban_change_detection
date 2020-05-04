# general modules
import json
import sys
import os
import numpy as np

# learning framework
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
from torchvision import transforms

# config for experiments
from experiment_manager import args
from experiment_manager.config import config

# custom stuff
import augmentations as aug
import evaluation_metrics as eval
import loss_functions as lf
import datasets

# networks from papers and ours
from networks import daudtetal2018
from networks import ours

# logging
import wandb


def setup(args):
    cfg = config.new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    return cfg


def train(net, cfg):

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.0005)

    # loss functions
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'WeightedBCEWithLogitsLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = lf.soft_dice_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = lf.soft_dice_loss_balanced
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = lf.jaccard_like_loss
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + lf.soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedComboLoss':
        criterion = lambda pred, gts: 2 * F.binary_cross_entropy_with_logits(pred, gts) + lf.soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'FrankensteinLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + lf.jaccard_like_balanced_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedFrankensteinLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts, pos_weight=positive_weight) + 5 * lf.jaccard_like_balanced_loss(pred, gts)
    else:
        criterion = lf.soft_dice_loss

    # reset the generators
    dataset = datasets.OSCDDataset(cfg, 'train')
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    positive_pixels = 0
    pixels = 0
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS
    for epoch in range(epochs):

        loss_tracker = 0
        net.train()

        for i, batch in enumerate(dataloader):

            pre_img = batch['pre_img'].to(device)
            post_img = batch['post_img'].to(device)

            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = net(pre_img, post_img)

            loss = criterion(output, label)
            loss_tracker += loss.item()
            loss.backward()
            optimizer.step()

            positive_pixels += torch.sum(label).item()
            pixels += torch.numel(label)

            global_step += 1

        if epoch % 2 == 0:
            # evaluate model after every epoch
            print(f'epoch {epoch} / {cfg.TRAINER.EPOCHS}')
            print(f'loss {loss_tracker:.5f}')
            print(f'positive pixel ratio: {positive_pixels / pixels:.3f}')
            if not cfg.DEBUG:
                wandb.log({f'positive pixel ratio': positive_pixels / pixels})
            positive_pixels = 0
            pixels = 0
            model_eval(net, cfg, device, run_type='train', epoch=epoch, step=global_step)
            model_eval(net, cfg, device, run_type='test', epoch=epoch, step=global_step)



def model_eval(net, cfg, device, run_type, epoch, step):

    def evaluate(y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        precision = eval.precision(y_true, y_pred, dim=0)
        recall = eval.recall(y_true, y_pred, dim=0)
        f1 = eval.f1_score(y_true, y_pred, dim=0)

        precision = precision.item()
        recall = recall.item()
        f1 = f1.item()

        return f1, precision, recall

    dataset = datasets.OSCDDataset(cfg, run_type, no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    y_true_set, y_pred_set = [], []
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            pre_img = batch['pre_img'].to(device)
            post_img = batch['post_img'].to(device)
            y_true = batch['label'].to(device)

            y_pred = net(pre_img, post_img)
            y_pred = torch.sigmoid(y_pred)
            threshold = 0.5
            y_pred = torch.gt(y_pred, threshold).float()

            y_true_set.append(y_true.flatten())
            y_pred_set.append(y_pred.flatten())

    y_true_set = torch.cat(y_true_set)
    y_pred_set = torch.cat(y_pred_set)

    f1, precision, recall = evaluate(y_true_set, y_pred_set)
    print(f'{run_type} f1: {f1:.3f}; precision: {precision:.3f}; recall: {recall:.3f}')

    if not cfg.DEBUG:
        wandb.log({
            f'{run_type} f1': f1,
            f'{run_type} precision': precision,
            f'{run_type} recall': recall,
            'step': step,
            'epoch': epoch,
        })


if __name__ == '__main__':

    # setting up config based on parsed argument
    parser = args.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    # TODO: load network from config
    if cfg.MODEL.TYPE == 'daudt_unet':
        net = daudtetal2018.UNet(12, 1)
    else:
        net = ours.UNet(cfg)


    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_change_detection',
            tags=['run', 'change', 'detection', ],
        )

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        train(net, cfg)
    except KeyboardInterrupt:
        print('Training terminated')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


