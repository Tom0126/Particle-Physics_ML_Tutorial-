#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/21 16:29
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : train.py
# @Software: PyCharm

import os.path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import copy
from scipy.special import expit

np.set_printoptions(threshold=np.inf)

class XGBDT():

    def __init__(self, file_dir, features, label_column, signal_label, bkg_label_list, file_name, ckp_dir):

        self.file_dir= file_dir
        self.features=features
        self.label_column=label_column
        self.signal_label=signal_label
        self.bkg_label_list=bkg_label_list
        self.file_name=file_name
        self.ckp_dir=ckp_dir


        self.train_var=None
        self.validation_var = None

        self.label_dict={
            0:'Electron',
            1:'Pion',
        }

    def load(self):

        self.train_var=self.prepare_data(dir='Train')
        self.validation_var=self.prepare_data(dir='Test')

    def min_max_scale(self, values):
        min_value = np.amin(values)
        max_value = np.amax(values)
        scaled_values =(values-min_value)/(max_value-min_value)
        return scaled_values

    def prepare_data(self, dir):

        file_dir = os.path.join(self.file_dir, dir)
        var_path = os.path.join(file_dir, self.file_name)
        var = pd.read_csv(var_path, usecols=self.features+[self.label_column])

        cut = var[self.label_column] == self.signal_label

        for label in self.bkg_label_list:
            cut = np.logical_or(cut, var[self.label_column] == label)

        var=var[cut]

        var['raw_labels']=var[self.label_column].copy()

        var[self.label_column] = np.where(var[self.label_column] == self.signal_label, True, False)

        return var

    def head_info(self, n):

       print(self.train_var.head(n=n), self.validation_var.head(n=n))

    def fit(self, eta, max_depth, subsample, colsample_bytree, objective, eval_metric, num_trees):

        train = xgb.DMatrix(data=self.train_var[self.features], label=self.train_var[self.label_column],
                            missing=-999.0, feature_names=self.features)

        param = dict()

        # Booster parameters
        param['eta'] = eta  # learning rate
        param['max_depth'] = max_depth  # maximum depth of a tree
        param['subsample'] = subsample  # fraction of events to train tree on
        param['colsample_bytree'] = colsample_bytree  # fraction of features to train tree on

        # Learning task parameters
        param['objective'] = objective  # objective function
        param['eval_metric'] = eval_metric  # evaluation metric for cross validation
        param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]


        booster = xgb.train(param, train, num_boost_round=num_trees)

        if not os.path.exists(self.ckp_dir):
            os.mkdir(self.ckp_dir)

        booster.save_model(os.path.join(self.ckp_dir, 'model.json'))

    def eval(self, source, model_path=None):

        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        legend_dict={
            0: 'Electron',
            1: 'Pion',
        }

        color_dict = {
            0: 'blue',
            1: 'red',
        }

        if not os.path.exists(self.ckp_dir):
            os.mkdir(self.ckp_dir)

        validation = xgb.DMatrix(data=self.validation_var[self.features], label=self.validation_var[self.label_column],
                                 missing=-999.0, feature_names=self.features)


        model_path= os.path.join(self.ckp_dir, 'model.json') if model_path==None else model_path

        booster=xgb.Booster()
        booster.load_model(model_path)

        print(booster.eval(validation))

        var_ranking=dict(sorted(booster.get_score(importance_type='gain').items(), key=lambda item: item[1]
                                , reverse=True))
        var_dict=dict()
        for var, wei in var_ranking.items():
            print('{}: {}'.format(var, wei/sum(var_ranking.values())))
            var_dict[var]=[wei/sum(var_ranking.values())]
        var_dict=pd.DataFrame(var_dict)
        var_dict.to_csv(os.path.join(self.ckp_dir, 'var.csv'))

        predictions = self.min_max_scale(np.array(booster.predict(validation)))

        range=[np.amin(predictions), np.amax(predictions)]

        label_size = 18
        # plot signal and background separately
        fig=plt.figure(figsize=(8,7))
        ax=plt.gca()

        plt.text(0.05, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.05, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        n1, _, _=plt.hist(predictions[validation.get_label().astype(bool)],
                 bins=30,
                 histtype='step', color='red', label=self.label_dict.get(self.signal_label), alpha=1, log=True, linewidth=3,
                 range=range,
                 weights=np.ones(len(predictions[validation.get_label().astype(bool)])) / len(predictions[validation.get_label().astype(bool)]),
                 stacked=False,
                 hatch='//',
                 )

        n2,_,_=plt.hist(
                  predictions[~(validation.get_label().astype(bool))], bins=30,
                 histtype='step', color='black', label='Backgrounds',
                 alpha=1, log=True, linewidth=3,
                 range=range,
                 weights= np.ones(len(predictions[~(validation.get_label().astype(bool))])) / len(
                              predictions[~(validation.get_label().astype(bool))]),
                 stacked=False,
                 hatch= '\\\\'
                 )

        # plt.ylim(1e-4, 1)

        # make the plot readable
        plt.xlabel('BDT {} likelihood'.format((self.label_dict.get(self.signal_label)).lower()), fontsize=label_size)
        plt.ylabel('# [Normalized]', fontsize=label_size)

        round_1= lambda x:round(x,2)
        plt.xticks(list(map(round_1,np.linspace(np.amin(predictions), np.amax(predictions),11))), fontsize=label_size)
        plt.yticks(fontsize=label_size)
        plt.legend(bbox_to_anchor=(0.95, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size)
        plt.ylim(top=15*np.amax(np.concatenate([n1,n2])))
        plt.savefig(os.path.join(self.ckp_dir, '{}_outputs_bdt_{}.png'.format(self.label_dict.get(self.signal_label), source)))
        plt.show()
        plt.close(fig)

        # plot likelihood for each particle type

        raw_labels = self.validation_var['raw_labels']


        all_list= [self.signal_label]+ self.bkg_label_list
        legend_labels = [legend_dict.get(label) for label in all_list]

        colors = [color_dict.get(color) for color in all_list]

        label_size = 18



        scores_to_plot = [predictions[raw_labels == label] for label in all_list]

        weights = [np.ones(len(scores)) / len(raw_labels) for scores in scores_to_plot]

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()

        plt.text(0.05, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )



        plt.text(0.05, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        n_count = []
        hatch_list = ['//', '//', '\\\\']
        for score, label, color, weight, hatch in zip(scores_to_plot, legend_labels, colors, weights,
                                                      hatch_list):
            n, _, _ = plt.hist(score, bins=30, label=label, range=[0, 1], histtype='step',
                               color=color,
                               density=False, stacked=False, alpha=1, log=True, linewidth=3,
                               weights=weight,
                               hatch=hatch,
                               )
            n_count.append(n)

        plt.legend(loc='upper right', bbox_to_anchor=(0.92,0.98), fontsize=label_size)

        plt.xticks(np.linspace(0,1,11), fontsize=label_size)
        plt.yticks(fontsize=label_size)
        plt.xlabel('BDT {} likelihood'.format((self.label_dict.get(self.signal_label)).lower()), fontsize=label_size)
        plt.ylabel('# [Normalized]', fontsize=label_size)


        plt.ylim(top=20*np.amax(np.concatenate(n_count)))
        plt.savefig(
            os.path.join(self.ckp_dir, 'bdt_score_{}_{}'.format(self.label_dict.get(self.signal_label), source)))
        plt.show()
        plt.close(fig)

        # choose score cuts:
        cuts = np.linspace(min(predictions), max(predictions), 101)
        nsignal = np.zeros(len(cuts))
        nbackground = np.zeros(len(cuts))

        text_dict = {
            'mc': 'MC test set',
            'tb': 'Data test set'
        }

        approach_dict = {
            'mc': 'MC training approach',
            'tb': 'Data training approach'
        }


        for i, cut in enumerate(cuts):
            nsignal[i] = np.sum((predictions[validation.get_label()==1]>cut)!=0)
            nbackground[i] = np.sum((predictions[validation.get_label()==0]>cut)!=0)

        efficiency=nsignal / np.sum((validation.get_label()==1)!=0)
        bkg_rej_rate=1-nbackground/np.sum((validation.get_label()==0)!=0)
        bkg_rej_ratio = np.sum((validation.get_label() == 0) != 0)/nbackground

        # plot efficiency vs. bkg r (ROC curve)
        fig=plt.figure(figsize=(8,7))
        ax=plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(efficiency,
                 bkg_rej_rate, 'o-',markersize=5,
                 color='red', label=approach_dict.get(source))
        # make the plot readable
        plt.ylim(0.9, 1.02)
        plt.xlim(0.9, 1)
        plt.xticks(np.linspace(0.9, 1, 11), fontsize=label_size)
        plt.yticks(np.linspace(0.9, 1, 11), fontsize=label_size)

        plt.xlabel('Efficiency '+r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('Bkg. rejection '+r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size)
        plt.savefig(os.path.join(self.ckp_dir, 'roc.png'))
        plt.close(fig)

        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        # s_b_threshold
        fig = plt.figure(figsize=(8, 7))


        ax = fig.add_subplot(111)
        l1 = ax.plot(cuts[::5], efficiency[::5], 'o', label=self.label_dict.get(self.signal_label), color='red', markersize=6)
        ax2 = ax.twinx()


        l2 = ax2.plot(cuts[::5], bkg_rej_rate[::5], '^', label='Backgrounds', color='black', markersize=6)

        ax.set_xlabel('BDT {} likelihood threshold'.format((self.label_dict.get(self.signal_label)).lower()), fontsize=label_size)
        ax.set_ylabel('{} efficiency '.format(self.label_dict.get(self.signal_label)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        ax2.set_ylabel('Bkg. rejection ' + r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=label_size)

        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        # plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        ax.tick_params(labelsize=label_size, direction='in', length=5)
        ax2.tick_params(labelsize=label_size, direction='in', length=5)

        ax.set_xticks(np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 11),
                      list(map(round_1, np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 11))))
        ax.set_yticks(np.linspace(0.9, 1, 6))
        ax2.set_yticks(np.linspace(0.9, 1, 6))

        ax.set_xlim(cuts[0], cuts[-1])
        ax.set_ylim(0.9, 1.02)
        ax2.set_ylim(0.9, 1.02)

        plt.minorticks_on()

        ax.tick_params(which='minor', direction='in', length=3)
        ax2.tick_params(which='minor', direction='in', length=3)

        ax.set_xticks(np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 51), minor=True)
        ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)
        ax2.set_yticks(np.linspace(0.9, 1, 26), minor=True)

        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.9, 0.98), fontsize=label_size)

        # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
        plt.savefig(os.path.join(self.ckp_dir, '{}_threshold_bdt_{}.png'.format(self.label_dict.get(self.signal_label),source)))
        plt.close(fig)

        # s_b_ratio_threshold
        fig = plt.figure(figsize=(8, 7))

        ax = fig.add_subplot(111)
        l1 = ax.plot(cuts[::5], efficiency[::5], 'o', label=self.label_dict.get(self.signal_label), color='red', markersize=6)
        ax2 = ax.twinx()
        l2 = ax2.plot(cuts[::5], bkg_rej_ratio[::5], '^', label='Backgrounds', color='black', markersize=6)

        ax.set_xlabel('BDT {} likelihood threshold'.format((self.label_dict.get(self.signal_label)).lower()), fontsize=label_size)
        ax.set_ylabel('{} efficiency '.format(self.label_dict.get(self.signal_label)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size-2)
        ax2.set_ylabel('Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)', fontsize=label_size-2)

        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        # plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        ax.tick_params(labelsize=label_size-2, direction='in', length=5)
        ax2.tick_params(labelsize=label_size-2, direction='in', length=5)

        ax.set_xticks(np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 11),
                      list(map(round_1, np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 11))))
        ax.set_yticks(np.linspace(0.9, 1, 6))
        ax2.set_yticks(np.linspace(0.9, 1, 6))

        ax.set_xlim(cuts[0], cuts[-1])
        ax.set_ylim(0.9, 1.03)
        ax2.set_ylim(1, 8*np.amax(bkg_rej_ratio[::5][~np.isinf(bkg_rej_ratio[::5])]))

        plt.minorticks_on()

        ax.tick_params(which='minor', direction='in', length=3)
        # ax2.tick_params(which='minor', direction='in', length=3)

        ax.set_xticks(np.linspace(round_1(np.amin(predictions)), round_1(np.amax(predictions)), 51), minor=True)
        ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)
        # ax2.set_yticks(np.linspace(0.9, 1, 26), minor=True)

        ax2.set_yscale('log')

        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.95, 0.98), fontsize=label_size-2)

        # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
        plt.savefig(os.path.join(self.ckp_dir, '{}_threshold_bdt_ratio_{}.png'.format(self.label_dict.get(self.signal_label),source)))
        plt.close(fig)


        df=pd.DataFrame(
            {
                'predictions': predictions,
                'labels':validation.get_label(),
                'raw_labels':self.validation_var['raw_labels'],

            }
        )
        df.to_csv(os.path.join(self.ckp_dir, 'eval.csv'))




if __name__ == '__main__':


    paras = {
        'eta': 0.1,
        'max_depth': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'num_trees': 100,
    }


    bdt = XGBDT(
        file_dir='/lustre/collider/songsiyuan/CEPC/PID/Data/BDT_tutorial',
        features=['Shower_density', 'Shower_start', 'Shower_layer_ratio', 'Shower_length', 'Hits_no', 'Shower_radius'],
        label_column='Particle_label',
        signal_label=0,
        bkg_label_list=[1],
        file_name='bdt_var.csv',
        ckp_dir='./Checkpoint',

    )
    bdt.load()
    bdt.fit(**paras)
    bdt.eval(source='mc')


    pass
