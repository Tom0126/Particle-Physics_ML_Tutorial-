#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/14 19:55
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : calculate_bdt_variable.py
# @Software: PyCharm
import os
import glob
import numpy as np
import pandas as pd
import uproot
import argparse
import math


# np.set_printoptions(threshold=np.inf)

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]



class BDT_VAR():

    def __init__(self, imgs_path, labels_path):

        self.imgs_path=imgs_path
        self.labels_path=labels_path
        self.bdt_var = dict()
        self.imgs=None
        self.labels=None


        self.center_distance=8.5

        self.rms = lambda x: np.sqrt(np.mean(x * x))
        self.sqr = lambda x: x*x

    def load(self):

        self.imgs=np.load(self.imgs_path)
        if self.labels_path!=None:
            self.labels=np.load(self.labels_path)
            self.bdt_var['Particle_label'] = self.labels
    def to_csv(self, save_dir, file_name='bdt_var.csv'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, file_name)

        df = pd.DataFrame(self.bdt_var)
        df.to_csv(save_path, index=False)

    def to_root(self,save_dir, file_name='bdt_var.root', tree_name='Calib_Hit'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, file_name)

        file = uproot.recreate(save_path)
        file[tree_name] = self.bdt_var

    def filter_zero_counts(self):
        counts=np.count_nonzero(self.imgs, axis=(1,2,3))
        self.labels[counts==0]=-1

    def filter_noise(self):

        e_dep=np.sum(self.imgs, axis=(1,2,3))

        self.labels[np.logical_and(e_dep<10, e_dep>0)]=3

    def calculate_n_hit(self, grouping_cell_size, img):

        count=0

        for i in range(img.shape[-1]):

            layer=img[:,:,i]
            if np.count_nonzero(layer)>0:

                for j in range(0,img.shape[0]-grouping_cell_size+1, grouping_cell_size):

                    for k in range(img.shape[1] - grouping_cell_size + 1):

                        patch=layer[j:j+grouping_cell_size,k:k+grouping_cell_size]

                        if np.count_nonzero(patch)>0:

                            count+=1
            else:
                continue

        return count


    def get_fd(self, beta:int, alpha:list, fd_name:str):

        fd = list()

        for img in self.imgs:

            fd_one=[]

            img_hit_no=np.count_nonzero(img)
            if img_hit_no==0:
                fd.append(0)
                continue

            if beta == 1:
                n_b= img_hit_no

            else:
                n_b=self.calculate_n_hit(grouping_cell_size=beta, img=img)

            for a in alpha:

                n_a=self.calculate_n_hit(grouping_cell_size=a, img=img)

                fd_one.append(1+math.log(n_b/n_a)/math.log(a))

            fd.append(np.mean(np.array(fd_one)))

        self.bdt_var[fd_name]=np.array(fd)


    def get_e_dep(self):

        self.bdt_var['E_dep']=np.sum(self.imgs,axis=(1,2,3))

    def get_shower_start(self, pad_fired=4):

        shower_starts=42 * np.ones(len(self.imgs))
        hits_count_layer=np.count_nonzero(self.imgs, axis=(1,2))

        for i, counts in enumerate(hits_count_layer):
            for layer in range(38):
                if counts[layer]>pad_fired and counts[layer+1]>pad_fired and counts[layer+2]>pad_fired:
                    shower_starts[i]=layer
                    break

        self.bdt_var['Shower_start']=shower_starts

    def get_shower_end(self, pad_fired=2):

        shower_ends=42 * np.ones(len(self.imgs))
        hits_count_layer=np.count_nonzero(self.imgs, axis=(1,2))

        for i, counts in enumerate(hits_count_layer):
            for layer in range(int(self.bdt_var['Shower_start'][i]),39):
                if counts[layer]<=pad_fired and counts[layer+1]<=pad_fired:
                    shower_ends[i]=layer+1
                    break

        self.bdt_var['Shower_end']=shower_ends


    def get_shower_layer(self,cell_rms=1):

        shower_layer_num = np.zeros(len(self.imgs))

        for i, img in enumerate(self.imgs):
            hits_indexes = np.nonzero(img)

            for layer in np.unique(hits_indexes[2]):

                x_indexes = hits_indexes[0][hits_indexes[2] == layer]
                y_indexes = hits_indexes[1][hits_indexes[2] == layer]

                if self.rms(x_indexes - self.center_distance) > cell_rms and self.rms(
                        y_indexes - self.center_distance) > cell_rms:
                    shower_layer_num[i] += 1

        self.bdt_var['Shower_layer']=shower_layer_num

    def get_hit_layer_no(self):

        hits_count_layer = np.count_nonzero(self.imgs, axis=(1, 2))
        layers_fired = np.sum((hits_count_layer > 0) != 0, axis=1)

        self.bdt_var['layers_fired']=layers_fired

    def get_shower_layer_ratio(self):

        self.bdt_var['Shower_layer_ratio']= np.where(self.bdt_var['layers_fired']!=0,
                                                     self.bdt_var['Shower_layer']/self.bdt_var['layers_fired'], 0 )

    def get_shower_density(self, pad_size=3):

        density=np.zeros(len(self.imgs))

        for i, img in enumerate(self.imgs):
            hits_num=np.count_nonzero(img)

            if hits_num==0:
                continue
            else:
                neib_num = 0
                for layer in np.arange(40)[np.count_nonzero(img, axis=(0,1))>0]:
                    samp_layer=img[:,:,layer]

                    hits_indexes=np.nonzero(samp_layer)

                    samp_layer_pad=np.pad(samp_layer, (int((pad_size-1)/2),int((pad_size-1)/2)), 'wrap')

                    for x_index, y_index in zip(hits_indexes[0], hits_indexes[1]):
                        neib_num+=np.count_nonzero(samp_layer_pad[x_index:x_index+pad_size, y_index:y_index+pad_size])

                density[i]=neib_num/hits_num

        self.bdt_var['Shower_density'] = density

    def get_shower_length(self,pad_fired=4):

        if 'Shower_start' not in self.bdt_var.keys():
            self.get_shower_start(pad_fired=pad_fired)

        shower_starts=self.bdt_var['Shower_start']
        shower_length=np.zeros(len(self.imgs))

        assert len(shower_starts) == len(self.imgs)

        for i, img in enumerate(self.imgs):

            if shower_starts[i] > 40:
                shower_length[i]=42

            else:

                hits_indexes = np.nonzero(img)

                rms_ = []
                shower_layers=np.unique(hits_indexes[2])[np.unique(hits_indexes[2])>shower_starts[i]] # layers after shower start layer

                for layer in shower_layers:

                    x_indexes=hits_indexes[0][hits_indexes[2]==layer]
                    y_indexes=hits_indexes[1][hits_indexes[2]==layer]

                    rms_.append(self.rms(np.sqrt(self.sqr(x_indexes-self.center_distance)+self.sqr(y_indexes-self.center_distance))))

                shower_length[i]= shower_layers[np.argmax(rms_)]-shower_starts[i]

        self.bdt_var['Shower_length'] = shower_length

    def get_hits_no(self):

        self.bdt_var['Hits_no']= np.count_nonzero(self.imgs, axis=(1,2,3))

    def get_shower_radius(self):

        shower_radius=[]

        for i, img in enumerate(self.imgs):

            if np.sum(img)==0:
                shower_radius.append(0)
            else:
                hits_indexes = np.nonzero(img)

                x_indexes=hits_indexes[0]
                y_indexes=hits_indexes[1]

                dist_=np.sqrt(self.sqr(x_indexes-self.center_distance)+self.sqr(y_indexes-self.center_distance))

                shower_radius.append(self.rms(dist_))

        self.bdt_var['Shower_radius'] = np.array(shower_radius)

    def get_e_mean(self):

        self.bdt_var['E_mean']=np.where(self.bdt_var['Hits_no']>0, self.bdt_var['E_dep']/self.bdt_var['Hits_no'], 0)

    def get_x_width(self):

        x_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                x_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                x_indexes = hits_indexes[0]


                x_width.append(self.rms(x_indexes-self.center_distance))

        self.bdt_var['X_width'] = np.array(x_width)

    def get_y_width(self):

        y_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                y_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                y_indexes = hits_indexes[1]


                y_width.append(self.rms(y_indexes-self.center_distance))

        self.bdt_var['Y_width'] = np.array(y_width)

    def get_z_width(self):

        z_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                z_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                z_indexes = hits_indexes[2]


                z_width.append(self.rms(z_indexes))

        self.bdt_var['Z_width'] = np.array(z_width)





def main(imgs_path, labels_path, save_dir):

    bdt_var=BDT_VAR(imgs_path=imgs_path,
                    labels_path=labels_path,
                    )
    bdt_var.load()

    bdt_var.filter_zero_counts()
    bdt_var.filter_noise()

    bdt_var.get_shower_density()
    bdt_var.get_shower_start()
    bdt_var.get_shower_end()
    bdt_var.get_hit_layer_no()
    bdt_var.get_shower_layer()
    bdt_var.get_shower_layer_ratio()
    bdt_var.get_e_dep()
    bdt_var.get_shower_length()
    bdt_var.get_hits_no()
    bdt_var.get_shower_radius()
    bdt_var.get_e_mean()
    bdt_var.get_x_width()
    bdt_var.get_y_width()
    bdt_var.get_z_width()

    bdt_var.to_csv(save_dir=save_dir)


if __name__ == '__main__':



    dataset_dir='/lustre/collider/songsiyuan/CEPC/PID/Data/tutorial'
    bdt_dir = '/lustre/collider/songsiyuan/CEPC/PID/Data/BDT_tutorial'

    os.makedirs(bdt_dir, exist_ok=True)

    for dir in ['Train', 'Validation', 'Test']:

        save_dir_=os.path.join(bdt_dir, dir)
        os.makedirs(save_dir_, exist_ok=True)

        load_dir=os.path.join(dataset_dir, dir)

        main(
            imgs_path=os.path.join(load_dir, 'imgs.npy'),
            labels_path=os.path.join(load_dir, 'labels.npy'),
            save_dir=save_dir_
        )

    pass
