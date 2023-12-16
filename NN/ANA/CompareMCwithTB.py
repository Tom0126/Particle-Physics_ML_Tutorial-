#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/4 21:55
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : CompareMCwithTB.py
# @Software: PyCharm

from PIDEvaluation import LoadCalibRoot
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    data='/lustre/collider/songsiyuan/CEPC/PID/Calib/e+/AHCAL_Run131_20221024_032845.root'
    mc='/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/ahcal_e+_40GeV_2cm_10k.npy'

    tb_40=LoadCalibRoot(data)
    hit_energy = tb_40.readHitEnergy()
    tot_e = []
    for _ in hit_energy:
        tot_e.append(np.sum(_))
    data_tot_e = np.array(tot_e)

    mc_40=np.load(mc)
    mc_tot_e=np.sum(mc_40,axis=(1,2,3))

    plt.hist(data_tot_e,bins=50,label='data',density=True)
    plt.hist(mc_tot_e, bins=50, label='mc',density=True)
    plt.legend()
    plt.savefig('/lustre/collider/songsiyuan/CEPC/PID/Calib/datavsmc_40GeV.png')
    pass
