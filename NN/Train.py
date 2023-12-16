# -*- coding: utf-8 -*-
"""
# @file name  : Train.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 15:09:00
# @brief      : CEPC PID
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from Evaluate import evaluate, get_ann_info
from Net.lenet import LeNet_bn
from Net.resnet import ResNet, BasicBlock, Bottleneck, ResNet_Avg
from Config.config import parser
from Data import loader
import sys
import pandas as pd


hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = parser.parse_args()

# set hyper-parameters
MAX_EPOCH = args.n_epoch
BATCH_SIZE = args.batch_size
LR = args.learning_rate
log_interval = args.log_interval
val_interval = args.val_interval
NUM_WORKERS = args.num_workers
OPTIM = args.optim
N_CLASSES = args.n_classes
STD_STATIC = args.standardize_static
L_GAMMA = args.l_gamma
STEP_SIZE = args.step
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN = True  # TODO Check
EVAL = True  # TODO Check
ANN_INFO = True  # TODO Check


data_dir_dict = {
    2: '/lustre/collider/songsiyuan/CEPC/PID/Data/tutorial',
}
net_name = 'tutorial_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}'.format(
    MAX_EPOCH, LR, BATCH_SIZE, OPTIM, N_CLASSES, L_GAMMA, STEP_SIZE)
# net_name='test'

net_used = 'lenet'  # TODO check

net_info_dict = {
    'lenet': {
        'n_classes': N_CLASSES,
        'path': './Net/lenet.py'
    },
}

net_para_dict = {
    'lenet': {'classes': N_CLASSES},

}

net_dict = {'lenet': LeNet_bn,
            'resnet': ResNet,
            'resnet_avg': ResNet_Avg,}
# TODO -----------------------------------------------------------------------


root_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model'  # train.py's dir
ckp_dir = os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint', net_name)
if not os.path.exists(ckp_dir):
    os.mkdir(ckp_dir)

model_path = os.path.join(ckp_dir, 'net.pth')
loss_path = ckp_dir + '/loss.png'
par_path = ckp_dir + '/hyper_paras.txt'
net_info_path = ckp_dir + '/{}.txt'.format(net_used)

ann_threshold_lists = np.linspace(0, 0.99999, 10000)
effi_points = [0.90, 0.93, 0.95, 0.97, 0.99][::-1]
ann_signal_label_list = [0,1,2]

if __name__ == '__main__':


    # TODO ============================ step 1/5 data ============================

    # DataLoder
    img_train_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Train/imgs.npy')
    label_train_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Train/labels.npy')

    img_vali_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Validation/imgs.npy')
    label_vali_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Validation/labels.npy')


    loader_train = loader.data_loader(img_train_path,
                                      label_train_path,
                                      num_workers=NUM_WORKERS,
                                      batch_size=BATCH_SIZE)
    loader_vali = loader.data_loader(img_vali_path,
                                      label_vali_path,
                                      num_workers=NUM_WORKERS,
                                      batch_size=BATCH_SIZE)

    # TODO ============================ step 2/5 model ============================

    net = net_dict.get(net_used)
    net_paras = net_para_dict.get(net_used)
    net = net(**net_paras)
    net.initialize_weights()

    # TODO ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()

    # TODO ============================ step 4/5 optimizer ============================

    optimizer_dict = {
        'SGD': optim.SGD(net.parameters(), lr=LR, momentum=0.9),
        'Adam': optim.AdamW(net.parameters(), lr=LR, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
    }

    optimizer = optimizer_dict.get(OPTIM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=L_GAMMA)

    net.to(DEVICE)
    criterion.to(DEVICE)

    # TODO ============================ step 5/5 train ============================

    if TRAIN:

        train_curve = list()
        valid_curve = list()

        for epoch in range(MAX_EPOCH):

            loss_mean = 0.
            correct = 0.
            total = 0.

            net.train()
            for i, (inputs, labels) in enumerate(loader_train):

                # input configuration
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                outputs = net(inputs)

                # backward
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()

                # update weights
                optimizer.step()

                # analyze results
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).squeeze().sum().cpu().numpy()

                # print results
                loss_mean += loss.item()
                train_curve.append(loss.item())
                if (i + 1) % log_interval == 0:
                    loss_mean = loss_mean / log_interval
                    print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, MAX_EPOCH, i + 1, len(loader_train), loss_mean, correct / total))
                    loss_mean = 0.

            scheduler.step()  # renew LR

            # validate the model
            if (epoch + 1) % val_interval == 0:

                correct_val = 0.
                total_val = 0.
                loss_val = 0.
                net.eval()
                with torch.no_grad():
                    for j, (inputs, labels) in enumerate(loader_vali):
                        # input configuration
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                        loss_val += loss.item()

                    loss_val_epoch = loss_val / len(loader_vali)
                    valid_curve.append(loss_val_epoch)
                    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, MAX_EPOCH, j + 1, len(loader_vali), loss_val_epoch, correct_val / total_val))


        train_x = range(len(train_curve))
        train_y = train_curve

        train_iters = len(loader_train)
        valid_x = np.arange(1,
                            len(valid_curve) + 1) * train_iters * val_interval - 1  # valid records epochloss，need to be converted to iterations
        valid_y = valid_curve

        plt.plot(train_x, train_y, label='Train')
        plt.plot(valid_x, valid_y, label='Validation')

        plt.legend(loc='upper right')
        plt.ylabel('loss value')
        plt.xlabel('Iteration')
        plt.savefig(loss_path)

        # save loss
        df1 = pd.DataFrame({
            'train_x': train_x,
            'train_y': train_y,

        })
        df1.to_csv(os.path.join(ckp_dir, 'loss_train.csv'))

        df2 = pd.DataFrame({
            'valid_x': valid_x,
            'valid_y': valid_y
        })
        df2.to_csv(os.path.join(ckp_dir, 'loss_validation.csv'))

        # save model
        torch.save(net.state_dict(), model_path)

    if EVAL:
        # TODO============================ evaluate model ============================

        pid_threshold = 0
        combin_datasets_dir_dict = {
            2: data_dir_dict.get(N_CLASSES) + '/Validation',
            3: data_dir_dict.get(N_CLASSES) + '/Validation',
            4: data_dir_dict.get(N_CLASSES) + '/Validation', }

        sep_datasets_dir_dict = {4: 'None'}

        evaluate(root_path=ckp_dir,
                 n_classes=N_CLASSES,
                 net_used=net_used,
                 net_dict=net_dict,
                 net_para_dict=net_para_dict,
                 combin_datasets_dir_dict=combin_datasets_dir_dict,
                 fig_dir_name='Fig',
                 threshold=pid_threshold,
                 threshold_num=101,
                 data_type='mc')






pass
