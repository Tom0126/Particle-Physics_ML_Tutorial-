#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /sw/anaconda/3.8-2021.05/thisconda.sh
conda activate pytorch

python /home/songsiyuan/CEPC/PID/Model/ANA/PIDEvaluation.py
