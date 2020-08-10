# Imports

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr

# Models
from benchmark.unet import Unet
from benchmark.siamunet_conc import SiamUnet_conc
from benchmark.siamunet_diff import SiamUnet_diff
from benchmark.fresunet import FresUNet

# Functions
from benchmark.utils import *
from benchmark.dataloader import *

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
from pathlib import Path
import time
from itertools import chain
import time
import warnings
from pprint import pprint


# Global Variables' Definitions

PATH_TO_DATASET = Path('/storage/shafner/urban_change_detection/OSCD_dataset/')

IS_PROTOTYPE = False

FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure

NUM_WORKERS = 16
BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 2  # 50

L = 1024
N = 2

NORMALISE_IMGS = True

TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1

TYPE = 3  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

LOAD_TRAINED = False

DATA_AUG = True


def train(n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_change_accuracy = 0 * t
    epoch_train_nochange_accuracy = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t

    #     mean_acc = 0
    #     best_mean_acc = 0
    fm = 0
    best_fm = 0

    lss = 1000
    best_lss = 1000

    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        tot_count = 0
        tot_loss = 0
        tot_accurate = 0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        #         for batch_index, batch in enumerate(tqdm(data_loader)):
        for batch in train_loader:
            I1 = Variable(batch['I1'].float().cuda())
            I2 = Variable(batch['I2'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))

            optimizer.zero_grad()
            output = net(I1, I2)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec = test(train_dataset)
        epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_train_change_accuracy[epoch_index] = cl_acc[1]
        epoch_train_precision[epoch_index] = pr_rec[0]
        epoch_train_recall[epoch_index] = pr_rec[1]
        epoch_train_Fmeasure[epoch_index] = pr_rec[2]

        #         epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_dataset)
        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_dataset)
        epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_test_change_accuracy[epoch_index] = cl_acc[1]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]

        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1], label='Train loss')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        plt.legend(handles=[l1_1, l1_2])
        plt.grid()
        #         plt.gcf().gca().set_ylim(bottom = 0)
        plt.gcf().gca().set_xlim(left=0)
        plt.title('Loss')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1], label='Train accuracy')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        plt.legend(handles=[l2_1, l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=3)
        plt.clf()
        l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_nochange_accuracy[:epoch_index + 1],
                         label='Train accuracy: no change')
        l3_2, = plt.plot(t[:epoch_index + 1], epoch_train_change_accuracy[:epoch_index + 1],
                         label='Train accuracy: change')
        l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1],
                         label='Test accuracy: no change')
        l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1],
                         label='Test accuracy: change')
        plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy per class')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=4)
        plt.clf()
        l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Train precision')
        l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Train recall')
        l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Train Dice/F1')
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
        plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Precision, Recall and F-measure')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        #         mean_acc = (epoch_test_nochange_accuracy[epoch_index] + epoch_test_change_accuracy[epoch_index])/2
        #         if mean_acc > best_mean_acc:
        #             best_mean_acc = mean_acc
        #             save_str = 'net-best_epoch-' + str(epoch_index + 1) + '_acc-' + str(mean_acc) + '.pth.tar'
        #             torch.save(net.state_dict(), save_str)

        #         fm = pr_rec[2]
        fm = epoch_train_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            save_str = 'net-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(fm) + '.pth.tar'
            torch.save(net.state_dict(), save_str)

        lss = epoch_train_loss[epoch_index]
        if lss < best_lss:
            best_lss = lss
            save_str = 'net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(lss) + '.pth.tar'
            torch.save(net.state_dict(), save_str)

        #         print('Epoch loss: ' + str(tot_loss/tot_count))
        if save:
            im_format = 'png'
            #         im_format = 'eps'

            plt.figure(num=1)
            plt.savefig(net_name + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig(net_name + '-02-accuracy.' + im_format)

            plt.figure(num=3)
            plt.savefig(net_name + '-03-accuracy-per-class.' + im_format)

            plt.figure(num=4)
            plt.savefig(net_name + '-04-prec-rec-fmeas.' + im_format)

    out = {'train_loss': epoch_train_loss[-1],
           'train_accuracy': epoch_train_accuracy[-1],
           'train_nochange_accuracy': epoch_train_nochange_accuracy[-1],
           'train_change_accuracy': epoch_train_change_accuracy[-1],
           'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1],
           'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
           'test_change_accuracy': epoch_test_change_accuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    print(pr_rec)

    return out


def test(dset):
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

        steps0 = np.arange(0, s[0], ceil(s[0] / N))
        steps1 = np.arange(0, s[1], ceil(s[1] / N))
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

    return net_loss, net_accuracy, class_accuracy, pr_rec


def save_test_results(dset):
    for name in tqdm(dset.names):
        with warnings.catch_warnings():
            I1, I2, cm = dset.get_img(name)
            I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
            I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255 * cm, 255 * np.squeeze(predicted.cpu().numpy()), 255 * cm), 2)
            io.imsave(f'{net_name}-{name}.png', I)


if __name__ == '__main__':

    train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=True, stride=TRAIN_STRIDE)
    weights = torch.FloatTensor(train_dataset.weights).cuda()
    print(weights)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=False, stride=TRAIN_STRIDE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    if TYPE == 0:
        #     net, net_name = Unet(2*3, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 3, 2), 'FresUNet'
    elif TYPE == 1:
        #     net, net_name = Unet(2*4, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 4, 2), 'FresUNet'
    elif TYPE == 2:
        #     net, net_name = Unet(2*10, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 10, 2), 'FresUNet'
    elif TYPE == 3:
        #     net, net_name = Unet(2*13, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 13, 2), 'FresUNet'

    net.cuda()

    criterion = nn.NLLLoss(weight=weights)  # to be used with logsoftmax output

    print('Number of trainable parameters:', count_parameters(net))

    # net.load_state_dict(torch.load('net-best_epoch-1_fm-0.7394933126157746.pth.tar'))

    if LOAD_TRAINED:
        net.load_state_dict(torch.load('net_final.pth.tar'))
        print('LOAD OK')
    else:
        t_start = time.time()
        out_dic = train()
        t_end = time.time()
        print(out_dic)
        print('Elapsed time:')
        print(t_end - t_start)

    if not LOAD_TRAINED:
        torch.save(net.state_dict(), 'net_final.pth.tar')
        print('SAVE OK')

    t_start = time.time()
    # save_test_results(train_dataset)
    save_test_results(test_dataset)
    t_end = time.time()
    print('Elapsed time: {}'.format(t_end - t_start))

    results = test(test_dataset)
    pprint(results)