#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 13:49
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : calculate.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def cal_composition(file_path):
    df=pd.read_csv(file_path)
    column=list(df.columns)
    compositions=np.transpose(df.values)
    print('particle\t{}\t{}\t{}'.format('muon', 'electron', 'pion'))
    for i, col in enumerate(column):
        print("{}\t{}\t{}\t{}".format(col, round(100*(compositions[i,0]/np.sum(compositions[i, 0:3])), 2),
                                      round(100*(compositions[i,1]/np.sum(compositions[i, 0:3])), 2)
                                      ,round(100*(compositions[i,2]/np.sum(compositions[i, 0:3])), 2)))


if __name__ == '__main__':
    print('TB')
    cal_composition(file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2022_pi_beam_info/composition.csv')

    print('MC')
    cal_composition(
        file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0627_res18_epoch_200_lr_1e-05_batch64_optim_SGD_classes_4_ihep_mc_v1/ANA/2022_pi_beam_info/composition.csv')

    # print('MC')
    # cal_composition(
    #     file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0626_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1/ANA/2022_pi_beam_info/composition.csv')
    #
    print('MC')
    cal_composition(
        file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1/ANA/2022_pi_beam_info/composition.csv')

    pass
