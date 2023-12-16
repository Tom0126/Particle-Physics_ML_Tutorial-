#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 19:59
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : e_sigma.py
# @Software: PyCharm

import os.path
import pandas as pd
import numpy as np
import glob
import uproot
from collections import Counter

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]
def pick_e(file_pid_path, file_path, threshold,particle_pid):

    data = ReadRoot(file_path=file_path, tree_name='Calib_Hit', exp=['Hit_Energy'])
    hit_e = data.readBranch('Hit_Energy')

    branch_list = ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise']
    ann_pid=ReadRoot(file_path=file_pid_path,tree_name='Calib_Hit',exp=branch_list)

    ann_score = []
    for branch in branch_list:
        branch_tags_ = ann_pid.readBranch(branch)

        ann_score.append(branch_tags_)
    ann_score = np.transpose(np.vstack(ann_score))

    max_scores, max_labels = np.amax(ann_score, axis=1), np.argmax(ann_score, axis=1)

    threshold_cut = max_scores >= threshold
    max_labels_cut = max_labels==branch_list.index(particle_pid)



    assert len(hit_e) == len(ann_score)

    hit_e=hit_e[np.logical_and(threshold_cut, max_labels_cut)]
    tot_e=[]
    for _ in hit_e:
        tot_e.append(np.sum(_))
    return np.array(tot_e)

def read_tot_e(ann_root_dir, file_root_dir, save_dir, threshold, particle_pid,dir_list ):

    ''' read the PId picked signal tot_e'''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for dir in dir_list:
        tot_e_ = []
        file_dir=os.path.join(file_root_dir,dir)
        pid_dir=os.path.join(ann_root_dir,dir)

        if os.path.exists(pid_dir):

            for ann_file_path in glob.glob(os.path.join(pid_dir, '*ANN.root')):
                file_path=ann_file_path.replace(pid_dir,file_dir)
                file_path=file_path.replace('_ANN.root','.root')

                if os.path.exists(file_path):

                    tot_e_.append(pick_e(file_pid_path=ann_file_path,file_path=file_path,threshold=threshold,
                                         particle_pid=particle_pid))

            if len(tot_e_)>0:
                tot_e_=np.concatenate(tot_e_)

                np.save(os.path.join(save_dir,dir+'.npy'),tot_e_)

        else:
            continue

def read_tb_data_tot_e(file_root_dir, save_root_dir, dir_list):

    '''read the tot energy deposition distribution from tb data'''
    for dir in dir_list:
        target_ep_dir=os.path.join(file_root_dir,dir)
        if os.path.exists(target_ep_dir):
            tot_e_=[]

            for file_path in glob.glob(os.path.join(target_ep_dir,'*.root')):
                root_file=ReadRoot(file_path=file_path,tree_name='Calib_Hit',exp=['Hit_Energy'])
                hit_e=root_file.readBranch('Hit_Energy')

                for _ in hit_e:
                    tot_e_.append(np.sum(_))


            if not os.path.exists(save_root_dir):
                os.mkdir(save_root_dir)

            save_path_=os.path.join(save_root_dir,dir+'.npy')
            np.save(save_path_,np.array(tot_e_))

        else:
            continue

def get_selection_efficiency(ann_root_dir, save_dir, threshold, particle_pid,dir_list, tree_name='Calib_Hit'):


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    effi = dict()

    for dir in dir_list:


        ann_picked_num = 0
        total_num = 0

        pid_dir = os.path.join(ann_root_dir, dir)

        if os.path.exists(pid_dir):

            for ann_file_path in glob.glob(os.path.join(pid_dir, '*ANN.root')):


                ann_file=ReadRoot(file_path=ann_file_path, tree_name=tree_name, exp=[particle_pid])
                ann_results=ann_file.readBranch(particle_pid)
                ann_picked_num+=len(ann_results[ann_results>=threshold])
                total_num+=len(ann_results)

                ann_file=None
                ann_results=None

            effi[dir]=[ann_picked_num/total_num]

        else:
            continue

    save_path=os.path.join(save_dir,'selection_efficiency.csv')
    df=pd.DataFrame(effi)
    df.to_csv(save_path,index=False)

def read_fd_e_hit(ann_root_dir, fd_root_dir, save_dir, threshold, particle_pid,dir_list ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for dir in dir_list:

        file_dir=os.path.join(fd_root_dir,dir)
        pid_dir=os.path.join(ann_root_dir,dir)


        fd_signal=[]
        e_hit_signal=[]

        fd_bkg=[]
        e_hit_bkg=[]

        if os.path.exists(pid_dir):

            for ann_file_path in glob.glob(os.path.join(pid_dir, '*ANN.root')):
                fd_file_path=ann_file_path.replace(pid_dir,file_dir)
                fd_file_path=fd_file_path.replace('_ANN.root','.root')

                if os.path.exists(fd_file_path):

                    ann_file=ReadRoot(file_path=ann_file_path, exp=[particle_pid, 'Event_Num'], tree_name='Calib_Hit')
                    ann_predcits= ann_file.readBranch(particle_pid)
                    ann_event_num= ann_file.readBranch('Event_Num')

                    fd_file=ReadRoot(file_path=fd_file_path, tree_name='Calib_Hit', exp=['FD_2D', 'Event_Num', 'Hit_Mean_Energy'])
                    fd = fd_file.readBranch('FD_2D')
                    fd_e_hit=fd_file.readBranch('Hit_Mean_Energy')
                    fd_event_num=fd_file.readBranch('Event_Num')

                    assert np.array_equal(ann_event_num,fd_event_num)

                    cut=ann_predcits>=threshold

                    fd_signal.append(fd[cut])
                    e_hit_signal.append(fd_e_hit[cut])

                    fd_bkg.append(fd[~cut])
                    e_hit_bkg.append(fd_e_hit[~cut])




            if len(fd_signal)>0:

                fd_signal=np.concatenate(fd_signal)
                e_hit_signal=np.concatenate(e_hit_signal)

                fd_bkg=np.concatenate(fd_bkg)
                e_hit_bkg=np.concatenate(e_hit_bkg)



                np.save(os.path.join(save_dir,dir+'_fd_signal.npy'),np.vstack([e_hit_signal,
                                                                        fd_signal,

                                                                        ]))

                np.save(os.path.join(save_dir, dir + '_fd_bkg.npy'), np.vstack([e_hit_bkg,
                                                                                   fd_bkg,

                                                                                   ]))
            else:

                np.save(os.path.join(save_dir, dir + '_fd_signal.npy'), np.vstack([e_hit_signal[0],
                                                                            fd_signal[0],

                                                                            ]))

                np.save(os.path.join(save_dir, dir + '_fd_bkg.npy'), np.vstack([
                                                                            e_hit_bkg[0],
                                                                            fd_bkg[0],
                                                                            ]))


        else:
            continue

def read_composition(ann_root_dir, save_dir, dir_list, threshold, branch_list, tree_name='Calib_Hit'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    compositions=[]
    for dir in dir_list:

        composi_=np.zeros(len(branch_list))
        num_=0

        pid_dir = os.path.join(ann_root_dir, dir)

        if os.path.exists(pid_dir):

            for ann_file_path in glob.glob(os.path.join(pid_dir, '*ANN.root')):
                ann_file_ = ReadRoot(file_path=ann_file_path, tree_name=tree_name, exp=branch_list)
                pid_tags_=[]
                for branch in branch_list:
                    branch_tags_=ann_file_.readBranch(branch)

                    pid_tags_.append(branch_tags_)
                pid_tags_=np.transpose(np.vstack(pid_tags_))

                max_scores, max_labels=np.amax(pid_tags_,axis=1), np.argmax(pid_tags_, axis=1)

                threshold_cut=max_scores>=threshold
                max_labels=max_labels[threshold_cut]
                count=Counter(max_labels)
                num_+=len(max_scores)

                for key, value in count.items():
                    composi_[key]+=value

            compositions.append(composi_/num_)

        else:
            continue

    save_path = os.path.join(save_dir, 'composition.csv')
    df = pd.DataFrame(np.transpose(np.vstack(compositions)), columns=dir_list)
    df.to_csv(save_path, index=False)

if __name__ == '__main__':

    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V3',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_pi_plus',
    #     'threshold': 0.9,
    #     'particle_pid': 'ANN_pi_plus',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]]
    #
    # }
    #
    # read_tot_e(**para)
    #
    # read_tb_data_tot_e(file_root_dir=para['file_root_dir'],
    #                    save_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2022/v3',
    #                    dir_list=para['dir_list'])
    #
    # get_selection_efficiency(ann_root_dir=para['ann_root_dir'],
    #                          save_dir=para['save_dir'],
    #                          threshold=para['threshold'],
    #                          particle_pid=para['particle_pid'],
    #                          dir_list=para['dir_list'],
    #                          tree_name='Calib_Hit')
    #
    # read_composition(ann_root_dir=para['ann_root_dir'],
    #                  save_dir=para['save_dir'],
    #                  threshold=para['threshold'],
    #                  branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    #                  dir_list=para['dir_list'],
    #                  )
    #

    #

    #
    # get_selection_efficiency(ann_root_dir=para['ann_root_dir'],
    #                          save_dir=para['save_dir'],
    #                          threshold=para['threshold'],
    #                          particle_pid=para['particle_pid'],
    #                          dir_list=para['dir_list'],
    #                          tree_name='Calib_Hit')
    #
    # read_fd_e_hit(ann_root_dir=para['ann_root_dir'],
    #               fd_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/FD_xia/SPS_V4/HCAL_Alone/0.5MIP_Energy_Cut/pi-_V4',
    #               save_dir=para['save_dir'],
    #               threshold=para['threshold'],
    #               particle_pid=para['particle_pid'],
    #               dir_list=para['dir_list'],)
    #
    #
    # read_composition(ann_root_dir=para['ann_root_dir'],
    #                  save_dir=para['save_dir'],
    #                  threshold=para['threshold'],
    #                  branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    #                  dir_list=para['dir_list'],
    #                  )

    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_e_plus',
    #     'threshold': 0.9,
    #     'particle_pid': 'ANN_e_plus',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]]
    #
    # }
    # read_tb_data_tot_e(file_root_dir=para['file_root_dir'],
    #            save_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2022/v1',
    #                    dir_list=para['dir_list'])
    #
    #
    # read_tot_e(**para)
    #
    #
    #
    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_mu_plus',
    #     'threshold': 0.9,
    #     'particle_pid': 'ANN_mu_plus',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]]
    #
    # }
    # #
    # read_tot_e(**para)
    #
    #
    #
    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_noise_2022',
    #     'threshold': 0.9,
    #     'particle_pid': 'ANN_noise',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]]
    #
    # }
    # #
    # read_tot_e(**para)
    # ckp_name='0615_res_epoch_200_lr_0.001_batch32_optim_SGD_classes_4_ihep_v1'
    # for threshold in np.arange(0.1,1,0.1):
    #
    #     para = {
    #         'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/{}'.format(ckp_name),
    #         'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/normal/pi-_V4.1',
    #         'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/ann_picked_mu_minus_{}'.format(ckp_name,int(100*threshold)),
    #         'threshold': threshold,
    #         'particle_pid': 'ANN_mu_plus',
    #         'dir_list': [str(i) + 'GeV' for i in [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]]
    #
    #     }
    #     #
    #     read_tot_e(**para)

        # para = {
        #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
        #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/normal/pi-_V4.1',
        #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_e_minus_{}'.format(int(100*threshold)),
        #     'threshold': threshold,
        #     'particle_pid': 'ANN_e_plus',
        #     'dir_list': [str(i) + 'GeV' for i in [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]]
        #
        # }
        # #
        # read_tot_e(**para)
        #
        # para = {
        #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
        #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/normal/pi-_V4.1',
        #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_pi_minus_{}'.format(int(100*threshold)),
        #     'threshold': threshold,
        #     'particle_pid': 'ANN_pi_plus',
        #     'dir_list': [str(i) + 'GeV' for i in [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]]
        #
        # }
        # #
        # read_tot_e(**para)
        #
        # para = {
        #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
        #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/normal/pi-_V4.1',
        #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_noise_2023_{}'.format(int(100*threshold)),
        #     'threshold': threshold,
        #     'particle_pid': 'ANN_noise',
        #     'dir_list': [str(i) + 'GeV' for i in [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]]
        #
        # }
        # #
        # read_tot_e(**para)

    # 2022
    threshold= 0.3

    para = {
        'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
        'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
        'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_mu_plus_{}'.format(int(100*threshold)),
        'threshold': threshold,
        'particle_pid': 'ANN_mu_plus',
        'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80,90, 100, 120]]

    }
    #
    # read_tot_e(**para)
    #
    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_e_plus_{}'.format(int(100*threshold)),
    #     'threshold': threshold,
    #     'particle_pid': 'ANN_e_plus',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80,90, 100, 120]]
    #
    # }
    # #
    # read_tot_e(**para)
    #
    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_pi_plus_{}'.format(int(100*threshold)),
    #     'threshold': threshold,
    #     'particle_pid': 'ANN_pi_plus',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80,90, 100, 120]]
    #
    # }
    # #
    # read_tot_e(**para)
    #
    # para = {
    #     'ann_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     'file_root_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V1',
    #     'save_dir': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_noise_2022_{}'.format(int(100*threshold)),
    #     'threshold': threshold,
    #     'particle_pid': 'ANN_noise',
    #     'dir_list': [str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80,90, 100, 120]]
    #
    # }
    # #
    # read_tot_e(**para)
    #
    # read_tb_data_tot_e(file_root_dir=para['file_root_dir'],
    #            save_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2022/v1',
    #                    dir_list=para['dir_list'])
    # read_composition(ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v3_2022/AHCAL_only/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #                  save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2022_pi_beam_info',
    #                  dir_list=[str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]],
    #                  threshold=0.9475494754947549,
    #                  branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    #                  )
    #
    # read_composition(
    #     ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v3_2022/AHCAL_only/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1/ANA/2022_pi_beam_info',
    #     dir_list=[str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]],
    #     threshold=0.9211692116921169,
    #     branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    #     )

    # read_composition(
    #     ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only_e_beam/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2023_e_beam_info',
    #     dir_list=[str(i) + 'GeV' for i in [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 250]],
    #     threshold=0.9475494754947549,
    #     branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    # )

    read_composition(
        ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_PS_pi_beam/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2023_pi_beam_info_ps',
        dir_list=[str(i) + 'GeV' for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]],
        threshold=0.9475494754947549,
        branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    )

    read_composition(
        ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_PS_e_beam/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2023_e_beam_info_ps',
        dir_list=[str(i) + 'GeV' for i in [1, 2, 3, 4, 5]],
        threshold=0.9475494754947549,
        branch_list=['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    )
    pass
