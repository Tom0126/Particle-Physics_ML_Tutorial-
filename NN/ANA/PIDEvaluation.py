#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/4 16:46
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : PIDEvaluation.py
# @Software: PyCharm

import uproot
import numpy as np
import matplotlib.pyplot as plt

class LoadCalibRoot():

    def __init__(self, file_path):
        self.path=file_path
        self.root_file=uproot.open(self.path)

    def readEnergyDeposit(self):
        tot_e=self.root_file['Energy deposition']
        tot_e=tot_e.to_numpy()
        return tot_e

    def readHitEnergy(self):
        calib_hit=self.readCalibHit()
        hit_energy=calib_hit['Hit_Energy']
        return hit_energy

    def readCalibHit(self):
        calib_hit=self.root_file['Calib_Hit']
        calib_hit=calib_hit.arrays(library="np")
        return calib_hit

    def readANNEPlus(self):
        calib_hit=self.readCalibHit()
        ann_e_plus=calib_hit['ANN_e_plus']
        return ann_e_plus

    def readANNMuPlus(self):
        calib_hit=self.readCalibHit()
        ann_mu_plus=calib_hit['ANN_mu_plus']
        return ann_mu_plus

    def readANNPiPlus(self):
        calib_hit=self.readCalibHit()
        ann_pi_plus=calib_hit['ANN_pi_plus']
        return ann_pi_plus


if __name__ == '__main__':
    run58_path='/lustre/collider/songsiyuan/CEPC/PID/Calib/pi+/40GeV/AHCAL_Run58_20221021_184832.root'
    run58_pid_path='/lustre/collider/songsiyuan/CEPC/PID/Calib/AHCAL_Run58_ANN_PID.root'

    run58=LoadCalibRoot(run58_path)
    ru658_pid=LoadCalibRoot(run58_pid_path)
    ann_pi_plus=ru658_pid.readANNPiPlus()
    hit_energy=run58.readHitEnergy()

    number=len(ann_pi_plus)

    tot_e=[]
    for _ in hit_energy:
        tot_e.append(np.sum(_))

    tot_e=np.array(tot_e)
    assert len(tot_e) == number

    tags_09=ann_pi_plus>0.9
    tags_075=ann_pi_plus>0.75
    tags_05 = ann_pi_plus > 0.5

    tot_e_09=tot_e[tags_09]
    tot_e_075 = tot_e[tags_075]
    tot_e_05 = tot_e[tags_05]

    r1=0
    r2=1500
    plt.figure(figsize=(6,5))
    plt.hist(tot_e, bins=50, label='Calib_Hit', histtype='step', range=[r1,r2],linewidth=2)
    plt.hist(tot_e_05,bins=50, label='Threshold: Prb>50%', histtype='step', range=[r1,r2],linewidth=2)
    plt.hist(tot_e_075, bins=50, label='Threshold: Prb>75%', histtype='step', range=[r1,r2],linewidth=2)
    plt.hist(tot_e_09, bins=50, label='Threshold: Prb>90%', histtype='step', range=[r1,r2],linewidth=2)
    plt.xlabel('[MeV]')

    plt.text(50, 24000, 'CEPC AHCAL PID', fontsize=15, fontstyle='oblique', fontweight='bold')
    plt.text(50, 22000, 'Run58 Pion+ @40GeV', fontsize=12, fontstyle='normal')
    plt.legend()
    plt.savefig('/lustre/collider/songsiyuan/CEPC/PID/Calib/AHCAL_Run58_ANN_PID.png')

    pass
