#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 14:43
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : compare_ann_bdt.py
# @Software: PyCharm

from e_sigma_reconstruct import read_ann_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import copy

class Compare():

    def __init__(self,bdt_eval_path, ann_scores_path, raw_labels_path, save_dir, ann_threshold_lists, bdt_threshold_lists,
                 ann_signal_label, source, n_classes, **kwargs):

        self.source=source
        self.save_dir=save_dir
        os.makedirs(self.save_dir,exist_ok=True)
        self.bdt_eval = pd.read_csv(bdt_eval_path)
        self.ann_scores= read_ann_score(ann_scores_path, n_classes,rt_df=False) if ann_scores_path !=None else kwargs.get('ann_scores')
        self.raw_labels = np.load(raw_labels_path) if raw_labels_path != None else kwargs.get(
            'raw_labels')

        self.ann_threshold_lists=ann_threshold_lists
        self.bdt_threshold_lists = bdt_threshold_lists
        self.ann_signal_label=ann_signal_label

        self.label_dict = {
            0: 'Muon',
            1: 'Electron',
            2: 'Pion',
        }

    def filter_label(self, label_list):

        self.label_list = label_list

        cut_ann=self.raw_labels==label_list[0]
        cut_bdt=self.bdt_eval['raw_labels']==label_list[0]


        for label in label_list:
            cut_ann=np.logical_or(cut_ann, self.raw_labels==label)
            cut_bdt = np.logical_or(cut_bdt, self.bdt_eval['raw_labels'] == label)

        self.ann_scores=self.ann_scores[cut_ann]
        self.raw_labels=self.raw_labels[cut_ann]
        self.bdt_eval=self.bdt_eval[cut_bdt]

    def get_ann_purity(self):


        signal_scores = self.ann_scores[:, self.ann_signal_label]

        purities = []

        for threshold in self.ann_threshold_lists:

            signal_picked = self.raw_labels[signal_scores >= threshold]

            purities.append(np.sum((signal_picked == self.ann_signal_label) != 0) / len(signal_picked))


        return np.array(purities)

    def get_ann_efficiency(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        efficiencies = []

        for threshold in self.ann_threshold_lists:

            signal_picked = self.raw_labels[signal_scores >= threshold]

            efficiencies.append(
                np.sum((signal_picked == self.ann_signal_label) != 0) /
                np.sum((self.raw_labels == self.ann_signal_label) != 0))



        return np.array(efficiencies)

    def get_ann_bkg_ratio(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_ratios = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            if bkg_picked_num > 0:
                bkg_ratios.append(np.sum((self.raw_labels != self.ann_signal_label) != 0) / bkg_picked_num)
            else:
                bkg_ratios.append(len(signal_scores))


        return np.array(bkg_ratios)

    def get_ann_bkg_rate(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_rates = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)


            bkg_rates.append(1-bkg_picked_num/np.sum((self.raw_labels != self.ann_signal_label) != 0))


        return np.array(bkg_rates)


    def get_ann_bkg_efficiency(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_efficiencies = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            bkg_efficiencies.append(bkg_picked_num / np.sum((self.raw_labels != self.ann_signal_label) != 0))

        return np.array(bkg_efficiencies)


    def get_significance(self):


        signal_scores = self.ann_scores[:, self.ann_signal_label]

        significances = []

        for threshold in self.ann_threshold_lists:

            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            if bkg_picked_num > 0:
                significances.append(np.sum((signal_picked == self.ann_signal_label) != 0) / math.sqrt(bkg_picked_num))
            else:
                significances.append(len(signal_scores))

        return np.array(significances)

    def get_bdt_ns_nb(self):

        ns_bdt=[]
        nb_bdt=[]

        predictions=self.bdt_eval['predictions'].values
        labels=self.bdt_eval['labels'].values
        for i, cut in enumerate(self.bdt_threshold_lists):
            ns_bdt.append(np.sum((predictions[labels==1]>=cut)!=0))
            nb_bdt.append( np.sum((predictions[labels==0]>=cut)!=0))

        self.ns_bdt=np.array(ns_bdt)
        self.nb_bdt=np.array(nb_bdt)

    def get_bdt_purity(self):
        return self.ns_bdt/ (self.ns_bdt+self.nb_bdt)

    def get_bdt_efficiency(self):
        return self.ns_bdt / np.sum((self.bdt_eval['labels'].values==1)!=0)

    def get_bdt_bkg_rate(self):
        return 1-self.nb_bdt/ np.sum((self.bdt_eval['labels'].values==0)!=0)

    def get_bdt_bkg_ratio(self):
        return np.sum((self.bdt_eval['labels'].values==0)!=0) / self.nb_bdt

    def get_bdt_bkg_efficiency(self):
        return self.nb_bdt/ np.sum((self.bdt_eval['labels'].values==0)!=0)

    def export_info(self):

        ann_name = 'ann_detailed_s_{}_b'.format(self.ann_signal_label)
        bdt_name = 'bdt_detailed_s_{}_b'.format(self.ann_signal_label)

        label_list = copy.deepcopy(self.label_list)
        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            ann_name = ann_name + '_' + str(b)
            bdt_name = bdt_name + '_' + str(b)

        self.df_ann=pd.DataFrame({
            'threshold':self.ann_threshold_lists,
            'ann_effi':self.get_ann_efficiency(),
            'ann_bkg_rate':self.get_ann_bkg_rate(),
            'ann_bkg_ratio': self.get_ann_bkg_ratio(),
            'ann_purity':self.get_ann_purity(),
            'ann_bkg_effi': self.get_ann_bkg_efficiency(),
        })

        self.df_ann.to_csv(os.path.join(self.save_dir, ann_name+'.csv'), index=False)

        self.df_bdt = pd.DataFrame({
            'threshold': self.bdt_threshold_lists,
            'bdt_effi': self.get_bdt_efficiency(),
            'bdt_bkg_rate': self.get_bdt_bkg_rate(),
            'bdt_bkg_ratio': self.get_bdt_bkg_ratio(),
            'bdt_purity': self.get_bdt_purity(),
            'bdt_bkg_effi': self.get_bdt_bkg_efficiency(),
        })

        self.df_bdt.to_csv(os.path.join(self.save_dir, bdt_name+'.csv'), index=False)

    def export_improvement_info(self, effi_points:list):
        '''change with increasing thresholds'''

        imp_dict=dict()


        ann_effi=[]
        bdt_effi=[]

        ann_puri=[]
        bdt_puri=[]

        ann_bkg_rej=[]
        bdt_bkg_rej=[]

        ann_bkg_ra = []
        bdt_bkg_ra= []

        ann_bkg_effi=[]
        bdt_bkg_effi=[]

        ann_start=0
        bdt_start=0

        ann_efficiency = self.get_ann_efficiency()
        ann_purity=self.get_ann_purity()
        ann_bkg_rate=self.get_ann_bkg_rate()
        ann_bkg_ratio = self.get_ann_bkg_ratio()
        ann_bkg_efficiency=self.get_ann_bkg_efficiency()

        bdt_efficiency = self.get_bdt_efficiency()
        bdt_purity = self.get_bdt_purity()
        bdt_bkg_rate = self.get_bdt_bkg_rate()
        bdt_bkg_ratio = self.get_bdt_bkg_ratio()
        bdt_bkg_efficiency = self.get_bdt_bkg_efficiency()

        for effi in effi_points:


            if ann_start >= len(ann_efficiency):
                ann_effi.append(ann_efficiency[-1])
                ann_puri.append(ann_purity[-1])
                ann_bkg_rej.append(ann_bkg_rate[-1])
                ann_bkg_effi.append(ann_bkg_efficiency[-1])
                ann_bkg_ra.append(ann_bkg_ratio[-1])



            for i, _ in enumerate(ann_efficiency[ann_start:]):

                if _ <= effi:
                    ann_effi.append(_)
                    ann_puri.append(ann_purity[ann_start:][i])
                    ann_bkg_rej.append(ann_bkg_rate[ann_start:][i])
                    ann_bkg_effi.append(ann_bkg_efficiency[ann_start:][i])
                    ann_bkg_ra.append(ann_bkg_ratio[ann_start:][i])
                    ann_start = ann_start + i + 1

                    break


            if bdt_start >= len(bdt_efficiency):
                bdt_effi.append(bdt_efficiency[-1])
                bdt_puri.append(bdt_purity[-1])
                bdt_bkg_rej.append(bdt_bkg_rate[-1])
                bdt_bkg_effi.append(bdt_bkg_efficiency[-1])
                bdt_bkg_ra.append(bdt_bkg_ratio[-1])

            for i, _ in enumerate(bdt_efficiency[bdt_start:]):
                if _ <= effi:
                    bdt_effi.append(_)
                    bdt_puri.append(bdt_purity[bdt_start:][i])
                    bdt_bkg_rej.append(bdt_bkg_rate[bdt_start:][i])
                    bdt_bkg_effi.append(bdt_bkg_efficiency[bdt_start:][i])
                    bdt_bkg_ra.append(bdt_bkg_ratio[bdt_start:][i])
                    bdt_start = bdt_start + i + 1

                    break

        imp_dict['ann_effi'] = np.around(np.array(ann_effi), decimals=3)[::-1]
        imp_dict['bdt_effi'] = np.around(np.array(bdt_effi), decimals=3)[::-1]

        imp_dict['bdt_puri']=np.around(np.array(bdt_puri),decimals=3)[::-1]
        imp_dict['ann_puri']=np.around(np.array(ann_puri),decimals=3)[::-1]
        imp_dict['puri_imp']=np.around(100*(np.array(ann_puri)-np.array(bdt_puri))/np.array(bdt_puri),decimals=3)[::-1]

        imp_dict['bdt_bkg_rej'] = np.around(np.array(bdt_bkg_rej),decimals=3)[::-1]
        imp_dict['ann_bkg_rej'] = np.around(np.array(ann_bkg_rej),decimals=3)[::-1]
        imp_dict['bkg_rej_imp'] = np.around(100 * (np.array(ann_bkg_rej) - np.array(bdt_bkg_rej)) / np.array(bdt_bkg_rej),decimals=3)[::-1]

        imp_dict['bdt_bkg_ra'] = np.around(np.array(bdt_bkg_ra), decimals=3)[::-1]
        imp_dict['ann_bkg_ra'] = np.around(np.array(ann_bkg_ra), decimals=3)[::-1]
        imp_dict['bkg_ra_imp'] = np.around(
            100 * (np.array(ann_bkg_ra) - np.array(bdt_bkg_ra)) / np.array(bdt_bkg_ra), decimals=3)[::-1]

        imp_dict['bdt_bkg_effi'] = np.around(1000*np.array(bdt_bkg_effi),decimals=3)[::-1]
        imp_dict['ann_bkg_effi'] = np.around(1000*np.array(ann_bkg_effi),decimals=3)[::-1]
        imp_dict['bkg_effi_imp'] = np.around(100 * (np.array(ann_bkg_effi) - np.array(bdt_bkg_effi)) / np.array(bdt_bkg_effi),decimals=3)[::-1]

        imp_name = 'ann_info_s_{}_b'.format(self.ann_signal_label)

        label_list=copy.deepcopy(self.label_list)
        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            imp_name = imp_name + '_' + str(b)

        self.improvement=pd.DataFrame(imp_dict, index=np.array(effi_points)[::-1])
        self.improvement.to_csv(os.path.join(self.save_dir, imp_name+'.csv'), index=True)



    def plot_purity_compare(self, y_ll=0, y_ul=1,x_ul=1):
        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_purity(), '^-', markersize=6,
                 color='red', label='ANN')

        plt.plot(self.get_bdt_efficiency(),
                 self.get_bdt_purity(), 'o-', markersize=6,
                 color='blue', label='BDT')

        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)
        plt.ylim(y_ll, y_ul+0.3*(y_ul-y_ll))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11-100*(1-x_ul))), fontsize=label_size)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)

        plt.minorticks_on()
        plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500*(x_ul-0.9)+1)), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} Efficiency '.format(self.label_dict.get(self.ann_signal_label))+r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('{} purity '.format(self.label_dict.get(self.ann_signal_label)) + r'$({N_{S}^{sel.}}/({N_{B}^{sel.}+N_{S}^{sel.}}))$', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, '{}_purity_effi_{}_compare.png'.format(self.label_dict.get(self.ann_signal_label),self.source)))
        plt.close(fig)

    def plot_bkg_rej_compare(self, y_ll=0, y_ul=1,x_ul=1):

        text_dict = {
            'mc': 'MC test set, MC training approach',
            'tb': 'Data test set, Data training approach'
        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=14, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_bkg_rate(), '^-', markersize=6,
                 color='red', label='ANN')

        plt.plot(self.get_bdt_efficiency(),
                 self.get_bdt_bkg_rate(), 'o-', markersize=6,
                 color='blue', label='BDT')

        # make the plot readable

        plt.tick_params(labelsize=14, direction='in', length=5)

        plt.ylim(y_ll, y_ul + 0.2 * (y_ul - y_ll))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=14)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=14,which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)


        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label))+r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
        plt.ylabel('Bkg. rejection rate '+r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=14)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=14)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'bkg_rej_effi_{}_compare.png'.format(self.source)))
        plt.close(fig)

    def plot_bkg_ratio_compare(self,y_ll=0, y_ul=1, x_ul=1):
        label_size = 18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.plot(self.get_ann_efficiency(),
                 self.get_ann_bkg_ratio(), '^-', markersize=6,
                 color='red', label='ANN')


        plt.plot(self.get_bdt_efficiency(),
                 self.get_bdt_bkg_ratio(), 'o-', markersize=6,
                 color='blue', label='BDT')
        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)

        plt.ylim(y_ll, y_ul)
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=label_size)
        # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)
        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('{} efficiency '.format(self.label_dict.get(self.ann_signal_label)) + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.9, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        fig_name = '{}_bkg_ratio_effi_{}_compare'.format(self.label_dict.get(self.ann_signal_label),self.source)
        label_list=copy.deepcopy(self.label_list)

        while self.ann_signal_label in label_list:
            label_list.remove(self.ann_signal_label)
        for b in label_list:
            fig_name = fig_name + '_' + str(b)


        plt.savefig(os.path.join(self.save_dir, fig_name+'.png'))
        plt.show()
        plt.close(fig)


def plot_ann_bdt_compare(
        ann_file_1:dict,
        ann_file_2:dict,
        ann_file_3:dict,
        bdt_file_1:dict,
        bdt_file_2:dict,
        bdt_file_3:dict,
        save_dir:str,
        fig_name:str,
        ann_x_var:str,
        ann_y_var:str,
        bdt_x_var:str,
        bdt_y_var:str,
        source:str,
        x_ll:float,
        x_ul:float,
        y_ll:float,
        y_ul:float,
        y_scale:str,
        x_label:str,
        y_label:str,
        legend_x:float,
        legend_y:float,
        line_width:float,


):
    ann_1=pd.read_csv(ann_file_1.get('path'), usecols=[ann_x_var,ann_y_var])
    ann_2 = pd.read_csv(ann_file_2.get('path'), usecols=[ann_x_var,ann_y_var])
    ann_3 = pd.read_csv(ann_file_3.get('path'), usecols=[ann_x_var,ann_y_var])
    bdt_1 = pd.read_csv(bdt_file_1.get('path'), usecols=[bdt_x_var,bdt_y_var])
    bdt_2 = pd.read_csv(bdt_file_2.get('path'), usecols=[bdt_x_var,bdt_y_var])
    bdt_3 = pd.read_csv(bdt_file_3.get('path'), usecols=[bdt_x_var,bdt_y_var])

    ann_x_1 = ann_1[ann_x_var]
    ann_x_2 = ann_2[ann_x_var]
    ann_x_3 = ann_3[ann_x_var]
    bdt_x_1 = bdt_1[bdt_x_var]
    bdt_x_2 = bdt_2[bdt_x_var]
    bdt_x_3 = bdt_3[bdt_x_var]

    ann_y_1 = ann_1[ann_y_var]
    ann_y_2 = ann_2[ann_y_var]
    ann_y_3 = ann_3[ann_y_var]
    bdt_y_1 = bdt_1[bdt_y_var]
    bdt_y_2 = bdt_2[bdt_y_var]
    bdt_y_3 = bdt_3[bdt_y_var]

    label_size = 18
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'tb': 'Data test set\nData training approach'
    }

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    plt.text(0.1, 0.89, text_dict.get(source), fontsize=label_size, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes,
             )

    plt.plot(ann_x_1,
             ann_y_1, ann_file_1.get('style'), markersize=6,
             color=ann_file_1.get('color'), label=ann_file_1.get('label'),
             linewidth=line_width,
             )

    plt.plot(ann_x_2,
             ann_y_2, ann_file_2.get('style'), markersize=6,
             color=ann_file_2.get('color'), label=ann_file_2.get('label'),
             linewidth=line_width,)

    plt.plot(ann_x_3,
             ann_y_3, ann_file_3.get('style'), markersize=6,
             color=ann_file_3.get('color'), label=ann_file_3.get('label'),
             linewidth=line_width,)

    plt.plot(bdt_x_1,
             bdt_y_1, bdt_file_1.get('style'), markersize=6,
             color=bdt_file_1.get('color'), label=bdt_file_1.get('label'),
             linewidth=line_width-1,
             )

    plt.plot(bdt_x_2,
             bdt_y_2, bdt_file_2.get('style'), markersize=6,
             color=bdt_file_2.get('color'), label=bdt_file_2.get('label'),
             linewidth=line_width-1,)

    plt.plot(bdt_x_3,
             bdt_y_3, bdt_file_3.get('style'), markersize=6,
             color=bdt_file_3.get('color'), label=bdt_file_3.get('label'),
             linewidth=line_width-1,)

    # make the plot readable

    plt.tick_params(labelsize=label_size, direction='in', length=5)

    plt.ylim(y_ll, y_ul)
    plt.xlim(x_ll, x_ul)
    plt.xticks(np.linspace(x_ll, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=label_size)


    plt.minorticks_on()
    plt.tick_params(labelsize=label_size, which='minor', direction='in', length=3)
    plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)

    plt.yscale(y_scale)
    if y_scale=='linear':
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=label_size)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

    plt.xlabel(xlabel=x_label,
               fontsize=label_size)
    plt.ylabel(ylabel=y_label,
               fontsize=label_size)

    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend(bbox_to_anchor=(legend_x, legend_y), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size - 4)

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.show()
    plt.close(fig)


def main_1():
    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1_2.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: \mu, bkg.:e, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: \mu, bkg.: e$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_2.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: \mu, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1_2.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: \mu, bkg.:e, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: \mu, bkg.: e$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_2.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: \mu, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1',
        fig_name='muon_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=30000,
        y_scale='log',
        x_label='Muon efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0_2.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: e, bkg.: \mu, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: e, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_2.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: e, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0_2.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: e, bkg.: \mu, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: e, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_2.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: e, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1',
        fig_name='electron_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=30000,
        y_scale='log',
        x_label='Electron efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0_1.csv',
            'style': '^-',
            'color': 'red',
            'label': r'$ANN, signal: \pi, bkg.: \mu, e$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0.csv',
            'style': '^-',
            'color': 'blue',
            'label': r'$ANN, signal: \pi, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_1.csv',
            'style': '^-',
            'color': 'green',
            'label': r'$ANN, signal: \pi, bkg.: e$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0_1.csv',
            'style': 'o--',
            'color': 'red',
            'label': r'$BDT, signal: \pi, bkg.: \mu, e$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0.csv',
            'style': 'o--',
            'color': 'blue',
            'label': r'$BDT, signal: \pi, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_1.csv',
            'style': 'o--',
            'color': 'green',
            'label': r'$BDT, signal: \pi, bkg.: e$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1',
        fig_name='pion_compare',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_ratio',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_ratio',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=1,
        y_ul=100000,
        y_scale='log',
        x_label='Pion efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)',
        legend_x=0.6,
        legend_y=0.5,
        line_width=4,
    )


def main_2():
    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1_2.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: \mu, bkg.:e, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_1.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: \mu, bkg.: e$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/ann_detailed_s_0_b_2.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: \mu, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1_2.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: \mu, bkg.:e, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_1.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: \mu, bkg.: e$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1/bdt_detailed_s_0_b_2.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: \mu, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_0_v1',
        fig_name='muon_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Muon efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.6,
        legend_y=0.4,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0_2.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: e, bkg.: \mu, \pi$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_0.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: e, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/ann_detailed_s_1_b_2.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: e, bkg.: \pi$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0_2.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: e, bkg.: \mu, \pi$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_0.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: e, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1/bdt_detailed_s_1_b_2.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: e, bkg.: \pi$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_1_v1',
        fig_name='electron_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Electron efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.6,
        legend_y=0.4,
        line_width=4,
    )

    plot_ann_bdt_compare(
        ann_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0_1.csv',
            'style': '-',
            'color': 'red',
            'label': r'$ANN, signal: \pi, bkg.: \mu, e$'},
        ann_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_0.csv',
            'style': '-',
            'color': 'blue',
            'label': r'$ANN, signal: \pi, bkg.: \mu$'},
        ann_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/ann_detailed_s_2_b_1.csv',
            'style': '-',
            'color': 'green',
            'label': r'$ANN, signal: \pi, bkg.: e$'},

        bdt_file_1={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0_1.csv',
            'style': '--',
            'color': 'red',
            'label': r'$BDT, signal: \pi, bkg.: \mu, e$'},
        bdt_file_2={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_0.csv',
            'style': '--',
            'color': 'blue',
            'label': r'$BDT, signal: \pi, bkg.: \mu$'},
        bdt_file_3={
            'path': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1/bdt_detailed_s_2_b_1.csv',
            'style': '--',
            'color': 'green',
            'label': r'$BDT, signal: \pi, bkg.: e$'},
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/compare_2_v1',
        fig_name='pion_compare_effi',
        ann_x_var='ann_effi',
        ann_y_var='ann_bkg_effi',
        bdt_x_var='bdt_effi',
        bdt_y_var='bdt_bkg_effi',
        source='mc',
        x_ll=0.9,
        x_ul=1,
        y_ll=0.000001,
        y_ul=1,
        y_scale='log',
        x_label='Pion efficiency' + r'$(N_{S}^{sel.}/N_{S})$',
        y_label='Bkg. efficiency ' + r'$(N_{B}^{sel.}/N_{B})$',
        legend_x=0.95,
        legend_y=0.4,
        line_width=4,
    )

def main_ana(ckp):
    num = 500
    for signal in range(3):
        cmp = Compare(
            bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
                signal),
            ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
            raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
            save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_{}_v1'.format(ckp,
                signal),
            ann_threshold_lists=np.linspace(0, 0.99999, 10000),
            bdt_threshold_lists=np.linspace(0, 1, 2000),
            ann_signal_label=signal,
            source='mc',
            n_classes=4
        )
        cmp.filter_label(label_list=[0, 1, 2])
        cmp.get_bdt_ns_nb()
        # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
        # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
        cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
        cmp.export_info()
        cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

        label_list = [0, 1, 2]
        label_list.remove(signal)

        for b in label_list:
            cmp = Compare(
                bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
                    signal),
                ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
                raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_{}_v1'.format(ckp,
                    signal),
                ann_threshold_lists=np.linspace(0, 0.99999, 10000),
                bdt_threshold_lists=np.linspace(0, 1, 2000),
                ann_signal_label=signal,
                source='mc',
                n_classes=4
            )
            cmp.filter_label(label_list=[signal, b])
            cmp.get_bdt_ns_nb()
            # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
            # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
            cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
            cmp.export_info()
            cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

def main_ana_e_pi(ckp):


    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_s_1_b_2_md_1000_nt_1000_var_12/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_1_v2'.format(ckp),
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 2000),
    #     ann_signal_label=1,
    #     source='mc',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[1, 2])
    # cmp.get_bdt_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

    cmp = Compare(
        bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_s_2_b_1_md_10_nt_10_var_12/eval.csv',
        ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/PIDTags/TV/imgs_ANN.root'.format(ckp),
        raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/{}/ANA/compare_2_v2'.format(ckp),
        ann_threshold_lists=np.linspace(0, 0.99999, 10000),
        bdt_threshold_lists=np.linspace(0, 1, 2000),
        ann_signal_label=2,
        source='mc',
        n_classes=4
    )
    cmp.filter_label(label_list=[1, 2])
    cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    cmp.export_info()
    cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])



if __name__ == '__main__':


    # TODO draft v1
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/Test/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='mc',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0,1,2])
    # cmp.get_bdt_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.92, 0.94, 0.96, 0.98, 0.99][::-1])
    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/Test/0720_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0720_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.92, 0.94, 0.96, 0.98, 0.99][::-1])

    # # TODO draft v1.3
    # num=500
    # for signal in range(3):
    #     cmp = Compare(
    #         bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(signal),
    #         ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/PIDTags/TV/imgs_ANN.root',
    #         raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/compare_{}_v3'.format(signal),
    #         ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #         bdt_threshold_lists=np.linspace(0, 1, 2000),
    #         ann_signal_label=signal,
    #         source='mc',
    #         n_classes=4
    #     )
    #     cmp.filter_label(label_list=[0,1,2])
    #     cmp.get_bdt_ns_nb()
    #     # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #     # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #     # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    #     cmp.export_info()
    #     cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])
    #
    #     label_list=[0,1,2]
    #     label_list.remove(signal)
    #
    #     for b in label_list:
    #         cmp = Compare(
    #             bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(
    #                 signal),
    #             ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/PIDTags/TV/imgs_ANN.root',
    #             raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
    #             save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.1_step_50_st_True_b_1_1_v1/ANA/compare_{}_v3'.format(
    #                 signal),
    #             ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #             bdt_threshold_lists=np.linspace(0, 1, 2000),
    #             ann_signal_label=signal,
    #             source='mc',
    #             n_classes=4
    #         )
    #         cmp.filter_label(label_list=[signal, b])
    #         cmp.get_bdt_ns_nb()
    #         # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #         # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    #         # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=30000)
    #         cmp.export_info()
    #         cmp.export_improvement_info(effi_points=[0.90, 0.93, 0.95, 0.97, 0.99][::-1])

    # TODO draft v1.4
    # main_ana(ckp='0901_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_1.0_step_100_st_True_b_1_1_f_k_3_f_s_2_f_p_3_v1')
    #
    # main_1()
    # main_2()



    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_md_100_nt_100_v3/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/TV/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/ANA/compare_v3',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 50000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=15000)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.90, 0.95, 0.99][::-1])

    # # TODO draft v2 no noise
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_no_noise_beam/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/Test/0728_tb_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_3_v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720_no_noise/Test/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_3_v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 1, 800),
    #     bdt_threshold_lists=np.linspace(0, 1, 500),
    #     ann_signal_label=2,
    #     n_classes=3,
    #     source='tb'
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.91, 0.93, 0.95, 0.97, 0.99][::-1])
    #
    # cmp = Compare(
    #     bdt_eval_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v4/eval.csv',
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_no_noise/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/ANA/compare',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 1000),
    #     ann_signal_label=2,
    #     n_classes=3,
    #     source='mc'
    # )
    # cmp.filter_label(label_list=[0, 1, 2])
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, y_ll=1, y_ul=10000)
    # cmp.export_info()
    # cmp.export_improvement_info(effi_points=[0.91, 0.93, 0.95, 0.97, 0.99][::-1])


    main_ana_e_pi(ckp='0901_mc_res18_avg_epoch_200_lr_0.0001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_10_st_True_b_1_1_f_k_5_f_s_1_f_p_2_v1')




    pass
