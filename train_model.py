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
import evaluation_metrics as metrics
import loss_functions as lf
import datasets

# all networks
from networks import network

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

    trfm = []
    trfm.append(aug.Numpy2Torch())
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        trfm.append(aug.UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        trfm.append(aug.ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    # TODO: separate for flip and rotate
    if cfg.AUGMENTATION.RANDOM_FLIP:
        trfm.append(aug.RandomFlip())
    if cfg.AUGMENTATION.RANDOM_ROTATE:
        trfm.append(aug.RandomRotate())
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
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    epochs = cfg.TRAINER.EPOCHS
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        epoch_loss = 0
        net.train()

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['img'].to(device)
            y_gts = batch['label'].to(device)

            y_pred = net(x)

            loss = criterion(y_pred, y_gts)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        # evaluate model after every epoch
        model_eval(net, cfg, device, run_type='test', epoch=epoch)
        model_eval(net, cfg, device, run_type='train', epoch=epoch)


def model_eval(net, cfg, device, run_type, epoch):


    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    dataset = datasets.OneraDataset(cfg, run_type)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': cfg.DATALOADER.NUM_WORKER,
        'shuffle': False,
        'drop_last': False,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    y_true_set, y_pred_set = [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            img = batch['img'].to(device)
            y_true = batch['label'].to(device)
            y_pred = net(img)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.gt(y_pred, 0.5).float()

            y_true_set.append(y_true)
            y_pred_set.append(y_pred)





    f1 = measurer.compute_f1()


    print(f'Computing {run_type} F1 score ', end=' ', flush=True)
    fpr, fnr = measurer.compute_basic_metrics()
    maxF1 = f1.max()
    argmaxF1 = f1.argmax()
    best_fpr = fpr[argmaxF1]
    best_fnr = fnr[argmaxF1]
    print(maxF1.item(), flush=True)

    set_name = 'test_set' if run_type == 'TEST' else 'training_set'
    wandb.log({f'{set_name} max F1': maxF1,
               f'{set_name} argmax F1': argmaxF1,
               # f'{set_name} Average Precision': ap,
               f'{set_name} false positive rate': best_fpr,
               f'{set_name} false negative rate': best_fnr,
               'step': step,
               'epoch': epoch,
               })




if __name__ == '__main__':

    # setting up config based on parsed argument
    parser = args.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    # TODO: load network from config
    net = network.U_Net(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, [1, 2])

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


