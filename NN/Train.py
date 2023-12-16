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
import SetSeed
import sys
from pid_batch import pid
from npy_pid_batch import npy_pid
import pandas as pd

from ANA.e_sigma import read_tot_e, read_tb_data_tot_e, read_fd_e_hit, read_composition
from ANA.e_sigma_reconstruct import plot_main, plot_tot_e_purifiled_compare, plot_fd_hit_e, plot_selection_efficiency, \
    plot_composition, plot_merged_evaluation, plot_merged_ann_score, plot_mc_tot_e_purifiled_compare

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = parser.parse_args()
if args.set_seed:
    SetSeed.setupSeed(args.seed)  # set random seed

# set hyper-parameters
MAX_EPOCH = args.n_epoch
BATCH_SIZE = args.batch_size
LR = args.learning_rate
log_interval = args.log_interval
val_interval = args.val_interval
NUM_WORKERS = args.num_workers
MEAN = args.mean
STD = args.std
OPTIM = args.optim
N_CLASSES = args.n_classes
STD_STATIC = args.standardize_static
L_GAMMA = args.l_gamma
STEP_SIZE = args.step
SHORT_CUT = bool(args.short_cut)
F_K=args.f_k
F_S=args.f_s
F_P=args.f_p


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# path
# TODO: data for this model

# TODO ---------------------check---------------------------------------------
TRAIN = True  # TODO Check
EVAL = True  # TODO Check
ANN_INFO = True  # TODO Check
PID_BEAM = False  # TODO Check

data_dir_dict = {
    2: '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/pi_pt_no_noise',
    3: '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_no_noise',
    4: '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_{}_{}'.format(args.b_xy, args.b_z),
}
net_name = '0901_mc_res18_avg_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_st_{}_b_{}_{}_f_k_{}_f_s_{}_f_p_{}_v1'.format(
    MAX_EPOCH, LR, BATCH_SIZE, OPTIM, N_CLASSES, L_GAMMA, STEP_SIZE, SHORT_CUT, args.b_xy, args.b_z, F_K, F_S, F_P)
# net_name='test'

net_used = 'resnet_avg'  # TODO check

net_info_dict = {
    'lenet': {
        'n_classes': N_CLASSES,
    },

    'resnet': {
        'block': 'Bottleneck',
        'layers': 'Res18',  # TODO check
        'num_classes': N_CLASSES,
        'start_planes': 40 // args.b_z,
        'short_cut': SHORT_CUT,
    },

    'resnet_avg': {
        'block': 'Bottleneck',
        'layers': 'Res18',  # TODO check
        'num_classes': N_CLASSES,
        'start_planes': 40 // args.b_z,
        'short_cut': SHORT_CUT,
    }



}

res_config_dict = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
    'Res18': [2, 2, 2, 2],
    'Res34': [3, 4, 6, 3],
    'Res101': [3, 4, 23, 3]
}

net_path_dict = {
    'lenet': '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/Net/lenet.py',
    'resnet': '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/Net/resnet.py',
    'resnet_avg': '/hpcfs/cepc/higgsgpu/siyuansong/PID/Model/Net/resnet.py',
}
net_para_dict = {
    'lenet': {'classes': N_CLASSES},

    'resnet':
        {'block': res_config_dict.get(net_info_dict['resnet'].get('block')),
         'layers': res_config_dict.get(net_info_dict['resnet'].get('layers')),
         'num_classes': net_info_dict['resnet'].get('num_classes'),
         'start_planes': net_info_dict['resnet'].get('start_planes'),
         'first_kernal': F_K,
         'first_stride': F_S,
         'first_padding': F_P,
         },

    'resnet_avg':
        {'block': res_config_dict.get(net_info_dict['resnet_avg'].get('block')),
         'layers': res_config_dict.get(net_info_dict['resnet_avg'].get('layers')),
         'num_classes': net_info_dict['resnet_avg'].get('num_classes'),
         'start_planes': net_info_dict['resnet_avg'].get('start_planes'),
         'first_kernal': F_K,
         'first_stride': F_S,
         'first_padding': F_P,
         }
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
    # save hyper-parameters
    dict = {'MAX_EPOCH': MAX_EPOCH, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'MEAN': MEAN, 'STD': STD, 'OPTIM': OPTIM
        , 'N_CLASSES': N_CLASSES, 'STD_STATIC': STD_STATIC, 'L_GAMMA': L_GAMMA, 'STEP': STEP_SIZE,
            'SHORT_CUT': SHORT_CUT, 'F_K':F_K}

    filename = open(par_path, 'w')  # dict to txt
    for k, v in dict.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()

    filename = open(net_info_path, 'w')  # dict to txt
    for k, v in (net_info_dict.get(net_used)).items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()

    os.system('cp {} {}'.format(net_path_dict.get(net_used), os.path.join(ckp_dir, 'net.py')))
    # TODO ============================ step 1/5 data ============================

    # DataLoder
    img_train_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Train/imgs.npy')
    label_train_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Train/labels.npy')

    img_valid_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Validation/imgs.npy')
    label_valid_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Validation/labels.npy')

    img_test_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Test/imgs.npy')
    label_test_path = os.path.join(data_dir_dict.get(N_CLASSES), 'Test/labels.npy')

    loader_train = loader.data_loader(img_train_path, label_train_path, mean=MEAN, std=STD,
                                      num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, mean_std_static=STD_STATIC)
    loader_valid = loader.data_loader(img_valid_path, label_valid_path, mean=MEAN, std=STD,
                                      num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, mean_std_static=STD_STATIC)
    # loader_test = loader.data_loader(img_test_path, label_test_path, num_workers=NUM_WORKERS)
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
                    for j, (inputs, labels) in enumerate(loader_valid):
                        # input configuration
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                        loss_val += loss.item()

                    loss_val_epoch = loss_val / len(loader_valid)
                    valid_curve.append(loss_val_epoch)
                    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, MAX_EPOCH, j + 1, len(loader_valid), loss_val_epoch, correct_val / total_val))

                # save CKP
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': net.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss,}, ck_path)

        train_x = range(len(train_curve))
        train_y = train_curve

        train_iters = len(loader_train)
        valid_x = np.arange(1,
                            len(valid_curve) + 1) * train_iters * val_interval - 1  # valid records epochloss，need to be converted to iterations
        valid_y = valid_curve

        plt.plot(train_x, train_y, label='Train')
        plt.plot(valid_x, valid_y, label='Valid')

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
                 mean=MEAN,
                 std=STD,
                 n_classes=N_CLASSES,
                 net_used=net_used,
                 net_dict=net_dict,
                 net_para_dict=net_para_dict,
                 combin_datasets_dir_dict=combin_datasets_dir_dict, fig_dir_name='Fig', threshold=pid_threshold,
                 threshold_num=101,
                 sep_datasets_dir_dict=sep_datasets_dir_dict, data_type='mc')



    if ANN_INFO:
        # TODO============================ ANN info ============================

        dataset_type = 'TV'

        ana_dir = os.path.join(ckp_dir, 'ANA')
        os.makedirs(ana_dir, exist_ok=True)

        pid_tag_dir = os.path.join(ana_dir, 'PIDTags')
        os.makedirs(pid_tag_dir, exist_ok=True)

        pid_tag_dir = os.path.join(pid_tag_dir, dataset_type)
        os.makedirs(pid_tag_dir, exist_ok=True)

        ana_scores_path = os.path.join(pid_tag_dir, 'imgs_ANN.root')

        get_ann_info(dataset_dir=os.path.join(data_dir_dict.get(N_CLASSES), dataset_type),
                     ana_scores_path=ana_scores_path,
                     ann_info_save_dir=ana_dir,
                     model_path=model_path,
                     n_classes=N_CLASSES,
                     net_used=net_used,
                     net_dict=net_dict,
                     net_para_dict=net_para_dict,
                     ann_threshold_lists=ann_threshold_lists,
                     ann_signal_label_list=ann_signal_label_list,
                     effi_points=effi_points
                     )


pass
