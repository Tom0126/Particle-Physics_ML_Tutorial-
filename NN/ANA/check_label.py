#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 12:44
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : check_label.py
# @Software: PyCharm



import numpy as np
from collections import Counter

def check_label(file_path):
    labels=np.load(file_path)
    count=Counter(labels)
    print(count)


if __name__ == '__main__':

    file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/AHCAL_Run258_20230506_141247/labels.npy'
    check_label(file_path=file_path)

    file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/AHCAL_Run261_20230507_015949/labels.npy'
    check_label(file_path=file_path)

    file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/AHCAL_Run276_20230507_144453/labels.npy'
    check_label(file_path=file_path)

    file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/AHCAL_Run277_20230507_151347/labels.npy'
    check_label(file_path=file_path)
    pass
