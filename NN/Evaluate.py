# -*- coding: utf-8 -*-
"""
# @file name  : Evaluate.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 12:49:00
# @brief      : CEPC PID
"""

import numpy as np
from Data import loader
import os
from torch.nn import Softmax
from ANA.acc import plotACCbar, plot_purity_threshold
from ANA.roc import plotROC, plot_s_b_threshold, plot_s_b_ratio_threshold
from torchmetrics.classification import MulticlassROC, MulticlassAUROC, MulticlassAccuracy
import torch


def purity_at_thresholds(model, dataloader, device, num_classes, thresholds_num=100):
    tps = np.zeros((num_classes, thresholds_num))
    nums = np.zeros((num_classes, thresholds_num))
    purities = np.zeros((num_classes, thresholds_num))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = Softmax(dim=1)(outputs)

            values, predicted = torch.max(outputs, 1)

            for t in range(thresholds_num):
                threshold = (t + 1) / float(thresholds_num)
                cut = values > threshold
                valid_preds = predicted[cut]
                valid_labels = labels[cut]
                for c in range(num_classes):
                    tps[c, t] += ((valid_preds == c) & (valid_labels == c)).cpu().float().sum().item()
                    nums[c, t] += (valid_preds == c).cpu().float().sum().item()

    for c in range(num_classes):
        for t in range(thresholds_num):
            purities[c, t] = tps[c, t] / nums[c, t] if nums[c, t] != 0 else 0

    # print(purities)
    return purities


def totalACC(data_loader, net, device):
    # evaluate
    correct_val = 0.
    total_val = 0.
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
        acc = "{:.2f}".format(100 * correct_val / total_val)
        # print("acc: {}%".format(acc))
        return float(acc)


def ACCParticle(data_loader, net, device, n_classes, threshold=0.9):
    # evaluate
    correct_val = np.zeros(n_classes)
    total_val = np.zeros(n_classes)

    predicts = []
    targets = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            #
            # for type in range(n_classes):
            #
            #     total_val[type] += (labels[labels==type]).size(0)
            #     correct_val[type] += (predicted[labels==type] == labels[labels==type]).squeeze().sum().cpu().numpy()

            # acc = 100 * correct_val / total_val

            predicts.append(outputs)
            targets.append(labels)
        targets = torch.cat(targets)
        predicts = torch.cat(predicts)

        mca = MulticlassAccuracy(num_classes=n_classes, average=None, threshold=threshold).to(device)
        acc = 100 * mca(predicts, targets).cpu().numpy()
        # print("acc: {}%".format(acc))
        return acc


def pbDisctuibution(data_loader, net, save_path, device):
    distributions = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

            # input configuration
            inputs = inputs.to(device)

            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                distributions = prbs.cpu().numpy()
            else:
                distributions = np.append(distributions, prbs.cpu().numpy(), axis=0)
        np.save(save_path, distributions)


def getROC(data_loader, net, device, save_path, num_class, ignore_index=None, threshold_num=21):
    preds = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                preds = prbs
                targets = labels
            else:
                preds = torch.cat((preds, prbs), 0)
                targets = torch.cat((targets, labels), 0)
        metric = MulticlassROC(num_classes=num_class, thresholds=threshold_num, ignore_index=ignore_index).to(device)
        fprs_, tprs_, thresholds_ = metric(preds, targets)
        fprs = []
        tprs = []
        for i, fpr in enumerate(fprs_):
            fprs.append(fpr.cpu().numpy())
            tprs.append(tprs_[i].cpu().numpy())

        np.array(fprs, dtype=object)
        np.array(tprs, dtype=object)
        np.save(save_path.format('fpr'), fprs)
        np.save(save_path.format('tpr'), tprs)

        mc_auroc = MulticlassAUROC(num_classes=num_class, average=None, thresholds=None, ignore_index=ignore_index)
        auroc = mc_auroc(preds, targets)
        np.save(save_path.format('auroc'), auroc.cpu().numpy())


def get_file_name(path):  # get .pth file
    image_files = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.pth':
            return file
    return None


def evaluate(root_path, n_classes,
             net_used,
             net_dict,
             net_para_dict,
             combin_datasets_dir_dict,
             data_type,
             fig_dir_name='Fig',
             threshold=0.9,
             threshold_num=21,
             comb_flag=True,
            ):


    # load model
    root_path = root_path
    model_path = os.path.join(root_path, get_file_name(root_path))

    ana_dir = os.path.join(root_path, 'ANA')
    if not os.path.exists(ana_dir):
        os.mkdir(ana_dir)

    fig_dir = os.path.join(ana_dir, fig_dir_name)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    save_combin_dir = os.path.join(ana_dir, 'combination')  # all test set combined
    if not os.path.exists(save_combin_dir):
        os.mkdir(save_combin_dir)

    save_combin_path = os.path.join(save_combin_dir, '{}.npy')  # store accuracy



    # TODO ---------------------------check-----------------------------------------------------------------------------

    signals_dict = {
        2: ['e+', 'pi+'],
       }
    # combination

    combin_datasets_dir = combin_datasets_dir_dict.get(n_classes)
    combin_datasets_path = os.path.join(combin_datasets_dir, 'imgs.npy')
    combin_labels_path = os.path.join(combin_datasets_dir, 'labels.npy')

    # roc

    save_roc_dir = os.path.join(ana_dir, 'roc')
    if not os.path.exists(save_roc_dir):
        os.mkdir(save_roc_dir)
    save_roc_path = os.path.join(save_roc_dir, '{}.npy')
    fpr_path = save_roc_path.format('fpr')
    tpr_path = save_roc_path.format('tpr')
    auroc_path = save_roc_path.format('auroc')

    net = net_dict.get(net_used)
    net_paras = net_para_dict.get(net_used)
    net = net(**net_paras)

    if torch.cuda.is_available():
        net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        device = 'cuda'
    else:
        device = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location=device))

    signals = signals_dict.get(n_classes)
    #  combination

    if comb_flag:
        # data loader

        loader_test = loader.data_loader(combin_datasets_path,
                                         combin_labels_path,
                                         mean_std_static=True,
                                         num_workers=0,
                                         batch_size=1000)

        acc = totalACC(loader_test, net, device)
        np.save(save_combin_path.format('combination'), np.array([acc]))

        acc_particles = ACCParticle(loader_test, net, device, n_classes, threshold=threshold)
        np.save(save_combin_path.format('acc_particles'), acc_particles)
        save_acc_particle_path = os.path.join(fig_dir, 'acc_particle.png')
        plotACCbar(acc_particles, save_acc_particle_path, threshold)


        # plot roc

        getROC(loader_test, net, device, save_roc_path, n_classes, threshold_num=threshold_num)

        save_roc_fig_path = os.path.join(fig_dir, '{}_roc.png')
        save_roc_threshold_path = os.path.join(fig_dir, '{}_threshold.png')
        save_roc_threshold_ratio_path = os.path.join(fig_dir, '{}_ratio_threshold_ann_'+data_type+'.png')

        for signal in signals:
            plotROC(fpr_path=fpr_path, tpr_path=tpr_path, auroc_path=auroc_path, signal=signal,
                    save_path=save_roc_fig_path.format(signal), data_type=data_type)
            plot_s_b_threshold(fpr_path=fpr_path, tpr_path=tpr_path, signal=signal,
                               save_path=save_roc_threshold_path.format(signal), threshold_num=threshold_num,
                               data_type=data_type)
            plot_s_b_ratio_threshold(fpr_path=fpr_path, tpr_path=tpr_path, signal=signal,
                                     save_path=save_roc_threshold_ratio_path,
                                     threshold_num=threshold_num,
                                     data_type=data_type)


