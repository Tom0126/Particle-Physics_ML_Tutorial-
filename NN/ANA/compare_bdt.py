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

class Compare():

    def __init__(self, bdt_eval_path:list, ann_scores_path, raw_labels_path, save_dir, ann_threshold_lists,
                 bdt_threshold_lists,
                 ann_signal_label, source, n_classes, **kwargs):

        self.source=source
        self.save_dir=save_dir
        self.eval_list=list()
        for path in bdt_eval_path:

            self.eval_list.append(pd.read_csv(path))

        self.ann_scores = read_ann_score(ann_scores_path, n_classes,
                                         rt_df=False) if ann_scores_path != None else kwargs.get('ann_scores')
        self.raw_labels = np.load(raw_labels_path) if raw_labels_path != None else kwargs.get(
            'raw_labels')
        self.ann_threshold_lists = ann_threshold_lists
        self.bdt_threshold_lists = bdt_threshold_lists
        self.ann_signal_label = ann_signal_label

        self.bdt_color=['#ff7f0e', '#2ca02c','#1f77b4', 'gold']

    def filter_label(self, label_list):

        self.label_list = label_list
        cut = self.raw_labels == label_list[0]

        for label in label_list:
            cut = np.logical_or(cut, self.raw_labels == label)

        self.ann_scores = self.ann_scores[cut]
        self.raw_labels = self.raw_labels[cut]
        self.eval_list= [bdt_eval[cut] for bdt_eval in self.eval_list]


    def export_info(self):

        self.df_ann = pd.DataFrame({
            'threshold': self.ann_threshold_lists,
            'ann_effi': self.get_ann_efficiency(),
            'ann_bkg_r': self.get_ann_bkg_rate(),
            'ann_purity': self.get_ann_purity(),
            'ann_bkg_ratio':self.get_ann_bkg_ratio(),
        })

        self.df_ann.to_csv(os.path.join(self.save_dir, 'ann_info.csv'), index=False)

        df_bdt=dict()
        df_bdt['threshold']=self.bdt_threshold_lists
        for i, effi in enumerate(self.get_bdt_efficiency()):
            df_bdt['bdt_effi_{}'.format(i)]=effi
        for i, bkg_r in enumerate(self.get_bdt_bkg_rate()):
            df_bdt['bdt_bkg_r_{}'.format(i)]=bkg_r
        for i, puri in enumerate(self.get_bdt_purity()):
            df_bdt['bdt_purity_{}'.format(i)]=puri
        for i, bkg_ra in enumerate(self.get_bdt_bkg_ratio()):
            df_bdt['bdt_bkg_ratio_{}'.format(i)]=bkg_ra

        self.df_bdt = pd.DataFrame(df_bdt)

        self.df_bdt.to_csv(os.path.join(self.save_dir, 'bdt_info.csv'), index=False)

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
                bkg_ratios.append(np.sum((self.raw_labels != self.ann_signal_label) != 0))

        return np.array(bkg_ratios)

    def get_ann_bkg_rate(self):

        signal_scores = self.ann_scores[:, self.ann_signal_label]

        bkg_rates = []

        for threshold in self.ann_threshold_lists:
            signal_picked = self.raw_labels[signal_scores >= threshold]
            bkg_picked_num = np.sum((signal_picked != self.ann_signal_label) != 0)

            bkg_rates.append(1 - bkg_picked_num / np.sum((self.raw_labels != self.ann_signal_label) != 0))

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

            # if bkg_picked_num > 0:
            significances.append(np.sum((signal_picked == self.ann_signal_label) != 0) / math.sqrt(bkg_picked_num))
            # else:
            #     significances.append(-1)

        return np.array(significances)

    def get_bdt_ns_nb(self):

        self.ns_bdt=list()
        self.nb_bdt=list()

        for eval in self.eval_list:
            predictions=eval['predictions'].values
            labels=eval['labels'].values
            ns_bdt = list()
            nb_bdt = list()

            for i, cut in enumerate(self.bdt_threshold_lists):
                ns_bdt.append(np.sum((predictions[labels==1]>=cut)!=0))
                nb_bdt.append(np.sum((predictions[labels==0]>=cut)!=0))

            self.ns_bdt.append(np.array(ns_bdt))
            self.nb_bdt.append(np.array(nb_bdt))



    def get_bdt_purity(self):

        purity=list()

        for ns , nb  in zip(self.ns_bdt, self.nb_bdt):

            purity.append(ns/(ns+nb))

        return np.vstack(purity)
    def get_bdt_efficiency(self):

        effi=list()

        for ns, eval in zip(self.ns_bdt, self.eval_list):

            effi.append(ns/np.sum((eval['labels'].values==1)!=0))

        return np.vstack(effi)

    def get_bdt_bkg_rate(self):
        bkg_r = list()

        for nb, eval in zip(self.nb_bdt, self.eval_list):
            bkg_r.append(1-nb / np.sum((eval['labels'].values == 0) != 0))

        return np.vstack(bkg_r)

    def get_bdt_bkg_efficiency(self):

        bkg_effi = list()

        for nb, eval in zip(self.nb_bdt, self.eval_list):
            bkg_effi.append(nb / np.sum((eval['labels'].values == 0) != 0))

        return np.vstack(bkg_effi)


    def get_bdt_bkg_ratio(self):

        bkg_ratio = list()

        for nb, eval in zip(self.nb_bdt, self.eval_list):
            bkg_ratio.append(np.sum((eval['labels'].values == 0) != 0)/nb)

        return np.vstack(bkg_ratio)



    def plot_purity_compare(self, labels,signal, y_ll=0, y_ul=1,x_ul=1, ):

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
                 self.get_ann_purity(), '^-', markersize=5,
                 color='red', label='ANN')

        bdt_effi=self.get_bdt_efficiency()
        bdt_purity=self.get_bdt_purity()

        for i, effi, puri, label in zip(range(len(labels)),bdt_effi, bdt_purity, labels):
            plt.plot(effi,
                    puri,'-', markersize=6,label=label, color=self.bdt_color[i]
                    )

        # make the plot readable

        plt.tick_params(labelsize=14, direction='in', length=5)
        plt.ylim(y_ll, y_ul+0.2*(y_ul-y_ll))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11-100*(1-x_ul))), fontsize=14)
        plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500*(x_ul-0.9)+1)), minor=True)
        plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('Pion Efficiency '+r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
        plt.ylabel('Pion purity ' + r'$({N_{S}^{sel.}}/({N_{B}^{sel.}+N_{S}^{sel.}}))$', fontsize=14)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes, loc='upper right', fontsize=14)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'purity_effi_{}_{}.png'.format(signal,self.source)))
        plt.close(fig)

    def plot_bkg_rej_compare(self,labels, y_ll=0, y_ul=1,x_ul=1):

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
                 self.get_ann_bkg_rate(), '^-', markersize=5,
                 color='red', label='ANN')

        bdt_effi = self.get_bdt_efficiency()
        bdt_bkg_r= self.get_bdt_bkg_rate()

        for i, effi, bkg_r, label in zip(range(len(labels)), bdt_effi, bdt_bkg_r, labels):
            plt.plot(effi,
                     bkg_r, 'o-', markersize=5, label=label, color=self.bdt_color[i]
                     )

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


        plt.xlabel('Pion efficiency '+r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
        plt.ylabel('Bkg. rejection rate '+r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=14)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes, loc='upper right', fontsize=14)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'bkg_rej_effi_{}.png'.format(self.source)))
        plt.close(fig)

    def plot_bkg_ratio_compare(self,labels, signal, y_ll=0, y_ul=1,x_ul=1,):

        label_size=18
        text_dict = {
            'mc': 'MC test set\nMC training approach',
            'tb': 'Data test set\nData training approach'
        }
        signal_dict={
            0: 'Muon',
            1: 'Electron',
            2: 'Pion',

        }

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        plt.text(0.1, 0.89, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        y_lim_list=[]
        # plt.plot(self.get_ann_efficiency(),
        #          self.get_ann_bkg_ratio(), '^-', markersize=5,
        #          color='red', label='ANN')

        bdt_effi = self.get_bdt_efficiency()
        bdt_bkg_r= self.get_bdt_bkg_ratio()

        for i, effi, bkg_r, label in zip(range(len(labels)),bdt_effi, bdt_bkg_r, labels):

            y_lim_list.append(bkg_r[effi>=0.9])
            plt.plot(effi,
                     bkg_r, '-', markersize=5, label=label, color=self.bdt_color[i], linewidth=4
                     )

        # make the plot readable

        plt.tick_params(labelsize=label_size, direction='in', length=5)


        plt.ylim(y_ll, 5*np.amax(np.concatenate(y_lim_list)))
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=label_size)
        # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=14,which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)
        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)


        plt.xlabel('{} efficiency '.format(signal_dict.get(signal))+r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size)
        plt.ylabel('Bkg. rejection '+r'$(N_{B}/N_{B}^{sel.}$)', fontsize=label_size)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.95, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size-2)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'bkg_ratio_effi_{}_{}.png'.format(signal,self.source)))
        plt.show()
        plt.close(fig)

    def plot_bkg_effi_compare(self, labels, y_ll=0, y_ul=1, x_ul=1):

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
                 self.get_ann_bkg_efficiency(), '^-', markersize=5,
                 color='red', label='ANN')

        bdt_effi = self.get_bdt_efficiency()
        bdt_bkg_r = self.get_bdt_bkg_efficiency()

        for i, effi, bkg_r, label in zip(range(len(labels)),bdt_effi, bdt_bkg_r, labels):
            plt.plot(effi,
                     bkg_r, 'o-', markersize=5, label=label, color=self.bdt_color[i]
                     )

        # make the plot readable

        plt.tick_params(labelsize=14, direction='in', length=5)

        # plt.ylim(y_ll, y_ul)
        plt.xlim(0.9, x_ul)
        plt.xticks(np.linspace(0.9, x_ul, round(11 - 100 * (1 - x_ul))), fontsize=14)
        # plt.yticks(np.linspace(y_ll, y_ul, 11), fontsize=14)

        plt.minorticks_on()
        plt.tick_params(labelsize=14, which='minor', direction='in', length=3)
        plt.xticks(np.linspace(0.9, x_ul, round(500 * (x_ul - 0.9) + 1)), minor=True)
        # plt.yticks(np.linspace(y_ll, y_ul, 51), minor=True)

        plt.xlabel('Pion efficiency ' + r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
        plt.ylabel('Bkg. rejection ' + r'$(N_{B}/N_{B}^{sel.}$)', fontsize=14)
        plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes, loc='upper right', fontsize=14)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, 'bkg_effi_{}.png'.format( self.source)))
        plt.close(fig)
if __name__ == '__main__':




    # TODO draft v1.1
    # cmp = Compare(
    #     bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_v2/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_v3/eval.csv'],
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/ANA/compare_v2',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='mc',
    #     n_classes=4
    # )
    #
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'],y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1,  labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'],y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1,  labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'], y_ll=1, y_ul=5000)
    # cmp.plot_bkg_effi_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'], y_ll=1, y_ul=5000)
    # cmp.export_info()
    #
    # cmp = Compare(
    #     bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_v2/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_v3/eval.csv'],
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/TV/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/ANA/compare_v2',
    #     ann_threshold_lists=np.linspace(0, 1, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 10000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    #
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, labels=[r'$\mathrm{BDT_{Data-6}}$', r'$\mathrm{BDT_{Data-8}}$', r'$\mathrm{BDT_{Data-12}}$'], y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, labels=[r'$\mathrm{BDT_{Data-6}}$', r'$\mathrm{BDT_{Data-8}}$', r'$\mathrm{BDT_{Data-12}}$'], y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, labels=[r'$\mathrm{BDT_{Data-6}}$', r'$\mathrm{BDT_{Data-8}}$', r'$\mathrm{BDT_{Data-12}}$'], y_ll=1, y_ul=5000)
    # cmp.plot_bkg_effi_compare(x_ul=1, labels=[r'$\mathrm{BDT_{Data-6}}$', r'$\mathrm{BDT_{Data-8}}$', r'$\mathrm{BDT_{Data-12}}$'], y_ll=1, y_ul=5000)
    # cmp.export_info()

    # TODO draft m, n 100
    # cmp = Compare(
    #     bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_md_100_nt_100_v1/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_md_100_nt_100_v2/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_md_100_nt_100_v3/eval.csv',
    #                    ],
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/ANA/compare_v2',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 100),
    #     bdt_threshold_lists=np.linspace(0, 1, 1000),
    #     ann_signal_label=2,
    #     source='mc',
    #     n_classes=4
    # )
    #
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1,
    #                         labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'],
    #                         y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1,
    #                          labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'],
    #                          y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1,
    #                            labels=[r'$\mathrm{BDT_{6}}$', r'$\mathrm{BDT_{8}}$', r'$\mathrm{BDT_{12}}$'],
    #                            y_ll=1, y_ul=5000)
    # cmp.plot_bkg_effi_compare(x_ul=1,
    #                           labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'],
    #                           y_ll=1, y_ul=5000)
    # cmp.export_info()

    # cmp = Compare(
    #     bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_md_100_nt_100_v1/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_md_100_nt_100_v2/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_ckv_fd_0720_beam_md_100_nt_100_v3/eval.csv',
    #                    ],
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/TV/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/ANA/compare_v2',
    #     ann_threshold_lists=np.linspace(0, 0.999, 1000),
    #     bdt_threshold_lists=np.linspace(0, 1, 1000),
    #     ann_signal_label=2,
    #     source='tb',
    #     n_classes=4
    # )
    #
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'], y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'], y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'], y_ll=1, y_ul=5000)
    # cmp.plot_bkg_effi_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$', r'$\mathrm{BDT_{MC-16}}$'], y_ll=1, y_ul=5000)
    # cmp.export_info()

    # # TODO draft m, n 10 mc no noise
    # cmp = Compare(
    #     bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v1/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v2/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v3/eval.csv',
    #                    '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_no_noise_mc_md_10_nt_10_v4/eval.csv',
    #                    ],
    #     ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/imgs_ANN.root',
    #     raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_no_noise/TV/labels.npy',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res34_epoch_200_lr_0.001_batch32_optim_Adam_classes_3_l_gamma_0.1_step_100v1/ANA/compare_v2',
    #     ann_threshold_lists=np.linspace(0, 0.99999, 10000),
    #     bdt_threshold_lists=np.linspace(0, 1, 1000),
    #     ann_signal_label=2,
    #     source='mc',
    #     n_classes=3
    # )
    #
    # cmp.get_bdt_ns_nb()
    # cmp.plot_purity_compare(x_ul=1,
    #                         labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$',
    #                                 r'$\mathrm{BDT_{MC-16}}$'],
    #                         y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_rej_compare(x_ul=1,
    #                          labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$',
    #                                  r'$\mathrm{BDT_{MC-16}}$'],
    #                          y_ll=0.9, y_ul=1)
    # cmp.plot_bkg_ratio_compare(x_ul=1,
    #                            labels=[r'$\mathrm{BDT_{6}}$', r'$\mathrm{BDT_{8}}$', r'$\mathrm{BDT_{12}}$', r'$\mathrm{BDT_{16}}$'],
    #                            y_ll=1, y_ul=5000)
    # cmp.plot_bkg_effi_compare(x_ul=1,
    #                           labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$',
    #                                   r'$\mathrm{BDT_{MC-16}}$'],
    #                           y_ll=1, y_ul=5000)
    # cmp.export_info()



    # TODO draft v1.3

    for signal in range(3):
        cmp = Compare(
            bdt_eval_path=['/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v1/eval.csv'.format(signal),
                           '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v2/eval.csv'.format(signal),
                           '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/bdt_0720_version_mc_signal_{}_md_100_nt_100_v3/eval.csv'.format(signal)],
            ann_scores_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/imgs_ANN.root',
            raw_labels_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
            save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_mc_res18_epoch_200_lr_0.0001_batch64_optim_SGD_classes_4_l_gamma_0.1_step_50v1/ANA/compare_signal_{}'.format(signal),
            ann_threshold_lists=np.linspace(0, 1, 2000),
            bdt_threshold_lists=np.linspace(0, 1, 2000),
            ann_signal_label=2,
            source='mc',
            n_classes=4
        )

        cmp.get_bdt_ns_nb()
        # cmp.plot_purity_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'],y_ll=0.9, y_ul=1)
        # cmp.plot_bkg_rej_compare(x_ul=1,  labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'],y_ll=0.9, y_ul=1)
        cmp.plot_bkg_ratio_compare(x_ul=1,  labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'], y_ll=10, y_ul=5000, signal=signal)
        # cmp.plot_bkg_effi_compare(x_ul=1, labels=[r'$\mathrm{BDT_{MC-6}}$', r'$\mathrm{BDT_{MC-8}}$', r'$\mathrm{BDT_{MC-12}}$'], y_ll=1, y_ul=5000)
        cmp.export_info()

    pass
