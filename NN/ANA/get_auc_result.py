#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 23:40
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : get_auc_result.py
# @Software: PyCharm

import glob
import numpy as np
if __name__ == '__main__':
    auc_dict=dict()
    for path in glob.glob('/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901*/ANA/roc/auroc.npy'):
        auc=np.load(path)

        auc_dict[list(path.split('/'))[-4]] ='{:.5f}'.format(np.mean(auc[1:3]))  # mean of auc
    auc_dict=sorted(auc_dict.items(), key=lambda x:x[1],reverse=True)
    for key, value in auc_dict:
        print(key, value)
    pass
