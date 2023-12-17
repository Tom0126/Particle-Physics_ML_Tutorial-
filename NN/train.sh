#! /bin/bash

source ../set_up.sh


# change hyper-parameters

n_epoch=2
batch_size=512
lr=0.0001
optim='SGD'
n_classes=2
l_gamma=0.1
step=1


python ./Train.py --n_epoch $n_epoch -b $batch_size -lr $lr --optim $optim --n_classes $n_classes --l_gamma $l_gamma --step $step
