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

# all networks
from networks import network, unet

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
    elif cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        balance_weight = [cfg.MODEL.NEGATIVE_WEIGHT, cfg.MODEL.POSITIVE_WEIGHT]
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=balance_weight)
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
    else:
        criterion = lf.soft_dice_loss

    # weight_tensor = torch.FloatTensor(2)
    # weight_tensor[0] = 0.20
    # weight_tensor[1] = 0.80
    # criterion = torch.nn.CrossEntropyLoss(weight_tensor.to(device)).to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = F.binary_cross_entropy_with_logits

    trfm = []
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        trfm.append(aug.UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        trfm.append(aug.ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    if cfg.AUGMENTATION.RANDOM_FLIP:
        trfm.append(aug.RandomFlip())
    if cfg.AUGMENTATION.RANDOM_ROTATE:
        trfm.append(aug.RandomRotate())
    trfm.append(aug.Numpy2Torch())
    trfm = transforms.Compose(trfm)

    # reset the generators
    dataset = datasets.OneraDataset(cfg, 'train', trfm)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': False,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    positive_pixels = 0
    pixels = 0
    epochs = cfg.TRAINER.EPOCHS
    for epoch in range(epochs):
        # print(f'Starting epoch {epoch}/{epochs}.')

        loss_tracker = 0
        net.train()

        for i, batch in enumerate(dataloader):

            pre_img = batch['pre_img'].to(device)
            post_img = batch['post_img'].to(device)

            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = net(pre_img, post_img)

            # loss = F.binary_cross_entropy_with_logits(output, label)
            loss = criterion(output, label)
            loss_tracker += loss.item()
            loss.backward()
            optimizer.step()

            # TODO: compute positive pixel ratio
            positive_pixels += torch.sum(label).item()
            pixels += torch.numel(label)

        if epoch % 10 == 0:
            # evaluate model after every epoch
            print(f'epoch {epoch} / {cfg.TRAINER.EPOCHS}')
            print(f'loss {loss_tracker:.5f}')
            print(f'positive pixel ratio: {positive_pixels / pixels:.3f}')
            positive_pixels = 0
            pixels = 0
            model_eval(net, cfg, device, run_type='train', epoch=epoch)
            # model_eval(net, cfg, device, run_type='test', epoch=epoch)


def model_eval(net, cfg, device, run_type, epoch):

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

    dataset = datasets.OneraDataset(cfg, run_type)
    dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
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


    # wandb.log({
    #     f'{run_type} F1': f1,
    #     'epoch': epoch,
    # })


if __name__ == '__main__':

    # setting up config based on parsed argument
    parser = args.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    # TODO: load network from config
    # net = network.U_Net(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, [1, 2])
    # net = network.U_Net(6, 1, [1, 2])
    net = unet.Unet(12, 1)

    # wandb.init(
    #     name=cfg.NAME,
    #     project='onera_change_detection',
    #     tags=['run', 'change', 'detection', ],
    # )

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


