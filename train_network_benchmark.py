# general modules
import json
import sys
import os
import numpy as np
from pathlib import Path
import math

# learning framework
import torch
from torch.utils import data as torch_data
from torch.autograd import Variable

# custom stuff
import evaluation_metrics as eval
import loss_functions as lf
import datasets
import utils

# networks from papers and ours
from benchmark.siamunet_diff import SiamUnet_diff
from benchmark.siamunet_conc import SiamUnet_conc
from benchmark.unet import Unet

# logging
import wandb

# Global Variables' Definitions

PATH_TO_DATASET = Path('/storage/shafner/urban_change_detection/OSCD_dataset/')

IS_PROTOTYPE = False

FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure

NUM_WORKERS = 16
BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 50  # 50

L = 1024
N = 2

TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1

DEBUG = False
SEED = 7


def train():

    t = np.linspace(1, N_EPOCHS, N_EPOCHS)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    for epoch_index in range(N_EPOCHS):
        print(f'epoch {epoch_index + 1} / {N_EPOCHS}')

        net.train()

        for batch in train_loader:
            t1_img = Variable(batch['t1_img'].float().cuda())
            t2_img = Variable(batch['t2_img'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))

            optimizer.zero_grad()
            output = net(t1_img, t2_img)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

        scheduler.step()

        test(test_dataset, 'test', epoch_index)
        test(train_dataset, 'train', epoch_index)


def test(dset, run_type: str, epoch: int):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for img_index in dset.names:
        I1_full, I2_full, cm_full = dset.get_img(img_index)

        s = cm_full.shape

        steps0 = np.arange(0, s[0], math.ceil(s[0] / N))
        steps1 = np.arange(0, s[1], math.ceil(s[1] / N))
        for ii in range(N):
            for jj in range(N):
                xmin = steps0[ii]
                if ii == N - 1:
                    xmax = s[0]
                else:
                    xmax = steps0[ii + 1]
                ymin = jj
                if jj == N - 1:
                    ymax = s[1]
                else:
                    ymax = steps1[jj + 1]
                I1 = I1_full[:, xmin:xmax, ymin:ymax]
                I2 = I2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
                I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()).cuda()

                output = net(I1, I2)
                loss = criterion(output, cm.long())
                #         print(loss)
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()

    net_loss = tot_loss / tot_count
    net_accuracy = 100 * (tp + tn) / tot_count

    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

    print(f'{run_type}: F1 score: {f_meas:.3f} - Precision: {prec:.3f} - Recall: {rec:.3f}')

    if not DEBUG:
        wandb.log({
            f'{run_type} F1 score': f_meas,
            f'{run_type} precision': prec,
            f'{run_type} recall': rec,
            f'{run_type} loss': net_loss,
            f'{run_type} accuracy': net_accuracy,
            'epoch': epoch,
        })




if __name__ == '__main__':

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = datasets.OSCDDatasetPaper(
        root_path=PATH_TO_DATASET,
        train=True,
        stride=TRAIN_STRIDE,
        augmentation=True,
        normalize=True,
        fp_modifier=FP_MODIFIER
    )
    weights = torch.FloatTensor(train_dataset.weights).cuda()
    print(weights)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = datasets.OSCDDatasetPaper(PATH_TO_DATASET, train=False, stride=TRAIN_STRIDE, augmentation=False)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    criterion = torch.nn.NLLLoss(weight=weights)

    # net, net_name = Unet(2*13, 2), 'FC-EF'
    # net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
    net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
    # net, net_name = FresUNet(2 * 13, 2), 'FresUNet'

    # tracking land with w&b
    if not DEBUG:
        wandb.init(
            name='benchmark',
            project='urban_change_detection_benchmark',
            tags=['run', 'change', 'detection', ],
        )

    try:
        train()
    except KeyboardInterrupt:
        print('Training terminated')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
