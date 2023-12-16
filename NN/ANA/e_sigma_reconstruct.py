#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 21:53
# @Author  : Tom SONG
# @Mail    : xdmyssy@gmail.com
# @File    : e_sigma_reconstruct.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import math
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import uproot
from matplotlib.ticker import ScalarFormatter
from collections import Counter

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]

def plot_tot_e(file_dir, save_path, func, ep_list, bins=100 ):
    fig = plt.figure(figsize=(24, 15))

    for i, ep in enumerate(ep_list):
        file_path = os.path.join(file_dir, str(ep) + 'GeV.npy')
        ed = np.load(file_path)

        up_limit = np.mean(ed) + 3 * np.std(ed)
        n, b = np.histogram(ed, bins=bins, range=[0, up_limit], density=True)

        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        popt, pcov = curve_fit(func, b, n, p0=[0.85, 4.2, b[np.argmax(n)], 10])
        plt.subplot(3, 4, i + 1)
        ax = plt.gca()

        plt.text(0.05, 0.9, 'CEPC Test Beam', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.text(0.05, 0.8, '{} data @{}GeV'.format(chr(960), ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.hist(ed, bins=bins, label='data', range=[0, up_limit], histtype='stepfilled', color='red', alpha=0.5,
                 density=True)

        fit_label = '{}: {}\n{}: {} \n{}: {} \n{}: {}'.format(chr(945), round(popt[0], 2), 'n', round(popt[1], 2), 'E',
                                                              round(popt[2], 2), chr(963), round(popt[3], 2))
        plt.plot(b, func(b, *popt), label=fit_label, linewidth=4, color='black')
        plt.legend(loc='center left')
        plt.ylabel('Normalized')
        plt.xlabel('[MeV]')



    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_tot_e_guass(file_dir, save_path, func, ep_list, bins=100, low_n_sigma_lists=None, up_n_sigma_lists=None, ):
    fig = plt.figure(figsize=(24, 15))

    for i, ep in enumerate(ep_list):
        file_path = os.path.join(file_dir, str(ep) + 'GeV.npy')

        ed = np.load(file_path)

        std = np.std(ed)
        mean = np.mean(ed)

        n, b = np.histogram(ed, bins=bins, range=[0, mean + 3 * std], density=True)
        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        up_limit = mean + up_n_sigma_lists[i] * std
        lower_limit = mean - low_n_sigma_lists[i] * std
        b2 = b[np.logical_and(b > lower_limit, b < up_limit)]
        n2 = n[np.logical_and(b > lower_limit, b < up_limit)]

        popt, pcov = curve_fit(func, b2, n2, p0=[1, 1.1 * b[np.argmax(n)], 10])

        plt.subplot(3, 4, i + 1)
        ax = plt.gca()

        plt.text(0.05, 0.9, 'CEPC Test Beam', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.text(0.05, 0.8, '{} data @{}GeV'.format(chr(960), ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.hist(ed, bins=bins, label='data', range=[0, mean + 3 * std], histtype='stepfilled', color='red', alpha=0.5,
                 density=True)

        fit_label = '{}: {} \n{}: {}'.format('E', round(popt[1], 2), chr(963), round(popt[2], 2))
        plt.plot(b2, func(b2, *popt), label=fit_label, linewidth=4, color='black')
        plt.legend(loc='center left')
        plt.ylabel('Normalized')
        plt.xlabel('[MeV)')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_resolution(para_dict, func, save_path, ep_list):
    x = list(para_dict.keys())
    y = []
    y_error=[]

    x_base = np.linspace(ep_list[0], ep_list[-1], 20)
    y_base = np.sqrt(np.square(0.6 / np.sqrt(x_base)) + 0.03**2)

    for value in para_dict.values():
        mean_= value[0][-2]
        std_=value[0][-1]
        num_=value[1]
        mean_error_=std_/math.sqrt(num_)
        std_error_=std_/math.sqrt(2*(num_-1))
        z_=std_ /mean_
        y.append(z_)
        z_error_=z_ * math.sqrt((std_error_ / std_)**2 + (mean_error_ / mean_)**2)
        y_error.append(z_error_)



    popt, pcov = curve_fit(func, x, y)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.text(0.15, 0.9, 'CEPC Test Beam', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.15, 0.8, '{} Resolution'.format(chr(960)), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )


    # plt.errorbar(x, y, y_error, linestyle='',capsize=3,label='Data', color='red')
    plt.plot(x, y, '.', label='Data', color='red')

    plt.plot(x_base, y_base, '--', label='Target: ' + r'$\frac{60\%}{\sqrt{E}}\oplus3\%$', color='green')

    if len(popt)==2:
        label='{} Fit: '.format(chr(960)) + r'$\frac{'+str(round(popt[0] * 100, 1))+'\%}{\sqrt{E}}\oplus'\
              +str(round(popt[1] * 100,1))+'\%$'
    else:
        label = '{} Fit: '.format(chr(960)) + r'$\frac{' + str(round(popt[0] * 100, 1)) + '\%}{\sqrt{E}}\oplus' + \
                r'\frac{' + str(round(popt[2] * 100, 1)) + '\%}{E}\oplus'\
                + str(round(popt[1] * 100, 1)) + '\%$'

    plt.plot(x_base, func(x_base, *popt),
             label=label, color='black')

    plt.legend(loc='center right')

    plt.xticks(ep_list)
    plt.xlabel('E [GeV]')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_linearity(para_dict, func, save_path, ep_list):
    x = list(para_dict.keys())
    y = []
    x_base = np.linspace(ep_list[0], ep_list[-1], 20)
    for value in para_dict.values():
        y.append(value[0][-2])

    popt, pcov = curve_fit(func, x, y)

    fig, axs = plt.subplots(2, 1, sharex='none', tight_layout=True, gridspec_kw={'height_ratios': [5, 2]},
                            figsize=(6, 8))

    axs = axs.flatten()

    axs[0].text(0.15, 0.9, 'CEPC Test Beam', fontsize=18, fontstyle='oblique', fontweight='bold',
                horizontalalignment='left',
                verticalalignment='center', transform=axs[0].transAxes, )
    axs[0].text(0.15, 0.8, '{} linearity'.format(chr(960)), fontsize=15, fontstyle='normal',
                horizontalalignment='left',
                verticalalignment='center', transform=axs[0].transAxes, )

    axs[0].plot(x, y, '.', label='data', color='red')
    axs[0].plot(x_base, func(x_base, *popt),
                label=('Fit E'), color='black')

    axs[0].set_xticks(ep_list)
    axs[0].set_xlabel('E [GeV]')
    axs[0].set_ylabel('Reconstructed E [MeV]')

    axs[1].plot(x, (np.array(y)-func(np.array(x), *popt)) / func(np.array(x), *popt)*100, color='black')
    axs[1].plot(x_base, 1.5 * np.ones(len(x_base)), '--', color='grey')
    axs[1].plot(x_base, -1.5 * np.ones(len(x_base)), '--', color='grey')
    axs[1].set_xticks(ep_list)
    axs[1].set_yticks(np.linspace(-3, 3, 13))
    axs[1].set_xlabel('E [GeV]')
    axs[1].set_ylabel(r'${(E-{E_{Fit}})}/{E_{Fit}}$'+ ' [%]')

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_tot_e_purifiled_compare(raw_file_dir, puri_file_dir_dict, save_path, ep_list, source, threshold,stacked,bins=100):


    color_dict={
        'muon_minus': 'green',
        'muon_plus': 'green',
        'electron':'blue',
        'positron': 'blue',
        'pion_minus':'red',
        'pion_plus': 'red',
        'noise':'orange',

    }

    greek_letter_dict={
        'muon_minus': r'$'+chr(956)+'^{-}$',
        'muon_plus': r'$'+chr(956)+'^{+}$',
        'electron': r'$e^{-}$',
        'positron': r'$e^{+}$',
        'pion_minus': r'$'+chr(960)+'^{-}$',
        'pion_plus': r'$'+chr(960)+'^{+}$',
        'noise': 'Noise',
    }

    fig = plt.figure(figsize=(24, 15))

    for i, ep in enumerate(ep_list):

        raw_file_path = os.path.join(raw_file_dir, str(ep) + 'GeV.npy')
        raw_ed_ = np.load(raw_file_path)
        up_limit = np.mean(raw_ed_) + 2.5 * np.std(raw_ed_)
        # low_limit=np.mean(puri_ed_) - 4 * np.std(puri_ed_)


        eds, labels, colors=[], [], []


        for key, puri_file_dir in puri_file_dir_dict.items():
            puri_file_path = os.path.join(puri_file_dir, str(ep) + 'GeV.npy')
            puri_ed_ = np.load(puri_file_path)
            eds.append(puri_ed_)
            labels.append(greek_letter_dict.get(key))
            colors.append(color_dict.get(key))



        plt.subplot(3, 4, i + 1)
        ax = plt.gca()

        plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.text(0.1, 0.8, source+' @{}GeV'.format(ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.text(0.1, 0.7, 'ANN threshold @{}'.format(threshold), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.hist(raw_ed_, bins=bins, label='{} beam'.format(greek_letter_dict.get(list(puri_file_dir_dict.keys())[0])), range=[0, up_limit], histtype='step',
                 color='black',
                 density=False, linewidth=1.5)



        plt.hist(eds, bins=bins, label=labels, range=[0, up_limit],
                 histtype='stepfilled', color=colors, alpha=0.5,
                 density=False,stacked=stacked)



        plt.legend(loc='upper right', bbox_to_anchor=(0.98,0.95))
        plt.xlabel('[MeV]')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def pick_npy_tot_e(file_pid_path, imgs_path, threshold):

    results=[]

    hit_e = np.sum(np.load(imgs_path),axis=(1,2,3))

    branch_list = ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise']
    ann_pid = ReadRoot(file_path=file_pid_path, tree_name='Calib_Hit', exp=branch_list)

    ann_score = []
    for branch in branch_list:
        branch_tags_ = ann_pid.readBranch(branch)
        ann_score.append(branch_tags_)

    ann_score = np.transpose(np.vstack(ann_score))

    max_scores, max_labels = np.amax(ann_score, axis=1), np.argmax(ann_score, axis=1)

    threshold_cut = max_scores >= threshold

    assert len(hit_e) == len(ann_score)

    for i in range(len(branch_list)):
        type_cut=max_labels==i

        results.append(hit_e[np.logical_and(threshold_cut,type_cut)])

    return results

def read_ann_score(file_pid_path, n_classes=4,rt_df=False):

    branch_list = ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise']
    branch_list=branch_list[:n_classes]

    ann_pid = ReadRoot(file_path=file_pid_path, tree_name='Calib_Hit', exp=branch_list)


    ann_score={}
    for branch in branch_list:
        ann_score[branch]=ann_pid.readBranch(branch)


    if rt_df:
        return pd.DataFrame(ann_score)
    else:
        return pd.DataFrame(ann_score).values




def get_purity(raw_label_path, ann_file_path, threshold_lists, signal_label,**kwargs):

    raw_labels = np.load(raw_label_path) if raw_label_path != None else kwargs.get('raw_labels')
    ann_scores = read_ann_score(file_pid_path=ann_file_path) if ann_file_path != None else kwargs.get('ann_scores')


    signal_scores=ann_scores[:, signal_label]

    purities=[]

    for threshold in threshold_lists:
        signal_picked=raw_labels[signal_scores>=threshold]

        purities.append(np.sum((signal_picked==signal_label)!=0)/len(signal_picked))

    # nodes=np.arange(0.1,1,0.1)
    # for node in nodes:
    #     for i, t_ in enumerate(threshold_lists):
    #         if t_>node:
    #             print('node: {} purities: {}'.format(t_, purities[i]))
    #             break

    return np.array(purities)


def get_efficiency(raw_label_path, ann_file_path, threshold_lists, signal_label, **kwargs):
    raw_labels = np.load(raw_label_path) if raw_label_path != None else kwargs.get('raw_labels')
    ann_scores = read_ann_score(file_pid_path=ann_file_path) if ann_file_path != None else kwargs.get('ann_scores')

    signal_scores = ann_scores[:, signal_label]


    efficiencies = []

    for threshold in threshold_lists:
        signal_picked = raw_labels[signal_scores >= threshold]

        efficiencies.append(np.sum((signal_picked == signal_label) != 0) / np.sum((raw_labels==signal_label)!=0))

    nodes=np.arange(0.1,1,0.1)
    # for node in nodes:
    #     for i, t_ in enumerate(threshold_lists):
    #         if t_>node:
    #             print('node: {} efficiencies: {}'.format(t_, efficiencies[i]))
    #             break

    return np.array(efficiencies)


def get_bkg_ratio(raw_label_path, ann_file_path, threshold_lists, signal_label, **kwargs):
    raw_labels = np.load(raw_label_path) if raw_label_path != None else kwargs.get('raw_labels')
    ann_scores = read_ann_score(file_pid_path=ann_file_path) if ann_file_path != None else kwargs.get('ann_scores')

    signal_scores = ann_scores[:, signal_label]

    bkg_ratios = []

    for threshold in threshold_lists:
        signal_picked = raw_labels[signal_scores >= threshold]
        bkg_picked_num=np.sum((signal_picked!=signal_label)!=0)

        if bkg_picked_num>0:
            bkg_ratios.append(np.sum((raw_labels != signal_label) != 0) / bkg_picked_num)
        else:
            bkg_ratios.append(-1)

    # nodes=np.arange(0.1,1,0.1)
    # for node in nodes:
    #     for i, t_ in enumerate(threshold_lists):
    #         if t_>node:
    #             print('node: {} bkgr: {}'.format(t_, bkg_ratios[i]))
    #             break

    return np.array(bkg_ratios)


def get_significance(raw_label_path, ann_file_path, threshold_lists, signal_label, **kwargs):

    raw_labels = np.load(raw_label_path) if raw_label_path != None else kwargs.get('raw_labels')
    ann_scores = read_ann_score(file_pid_path=ann_file_path) if ann_file_path != None else kwargs.get('ann_scores')

    signal_scores = ann_scores[:, signal_label]

    significances = []

    for threshold in threshold_lists:

        signal_picked = raw_labels[signal_scores >= threshold]
        bkg_picked_num = np.sum((signal_picked != signal_label) != 0)

        if bkg_picked_num > 0:
            significances.append(np.sum((signal_picked == signal_label) != 0) / math.sqrt(bkg_picked_num))
        else:
            significances.append(-1)


    return np.array(significances)

def plot_evaluation(raw_label_path, ann_file_path, label_dict, signal_label, threshold_lists,save_path,ep=None, **kwargs):



    greek_letter_dict = {
        'muon_minus': r'$' + chr(956) + '^{-}$',
        'muon_plus': r'$' + chr(956) + '^{+}$',
        'electron': r'$e^{-}$',
        'positron': r'$e^{+}$',
        'pion_minus': r'$' + chr(960) + '^{-}$',
        'pion_plus': r'$' + chr(960) + '^{+}$',
        'noise': 'Noise',
    }

    signal_greek = greek_letter_dict.get(label_dict[signal_label])
    linewidth=4

    purities=get_purity(raw_label_path=raw_label_path, ann_file_path=ann_file_path, threshold_lists=threshold_lists,
                        signal_label=signal_label,
                        raw_labels=kwargs.get('raw_labels'), ann_scores= kwargs.get('ann_scores'))

    efficiencies = get_efficiency(raw_label_path=raw_label_path, ann_file_path=ann_file_path, threshold_lists=threshold_lists,
                          signal_label=signal_label,
                          raw_labels=kwargs.get('raw_labels'), ann_scores=kwargs.get('ann_scores'))

    bkg_ratios = get_bkg_ratio(raw_label_path=raw_label_path, ann_file_path=ann_file_path, threshold_lists=threshold_lists,
                          signal_label=signal_label,
                          raw_labels=kwargs.get('raw_labels'), ann_scores=kwargs.get('ann_scores'))

    significances=get_significance(raw_label_path=raw_label_path, ann_file_path=ann_file_path, threshold_lists=threshold_lists,
                          signal_label=signal_label,
                          raw_labels=kwargs.get('raw_labels'), ann_scores=kwargs.get('ann_scores'))

    fig = plt.figure(figsize=(12, 30))

    plt.subplot(3,1,1)


    ax=plt.gca()
    ax_2=ax.twinx()

    ax.tick_params(labelsize=15)
    ax_2.tick_params(labelsize=15)


    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=25, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    tag = 'Simulation @{}GeV'.format(ep) if ep != None else 'Simulation'

    plt.text(0.1, 0.8, tag, fontsize=23, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes,)

    l1=ax.plot(threshold_lists, efficiencies, linewidth=linewidth, color='red',
            label='{} signal efficiency'.format(signal_greek))

    l2=ax_2.plot(threshold_lists[bkg_ratios!=-1], bkg_ratios[bkg_ratios!=-1], linewidth=linewidth, color='black',
              label='Background rejection rate' )

    lns=l1+l2
    if -1 in bkg_ratios:
        l3=ax_2.plot(threshold_lists[bkg_ratios == -1],
                  np.amax(bkg_ratios[bkg_ratios!=-1])*np.ones(np.sum((bkg_ratios == -1)!=0)),
                  '*', color='orange',
              label='Pure {}'.format(signal_greek))
        lns+=l3

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.9,0.95),fontsize=15)

    ax.set_xlabel('ANN probability  cut', fontsize=20)

    ax.set_ylabel('Efficiency '+r'$N_{S}^{sel}/N_{S}$', fontsize=20)
    ax_2.set_ylabel('Rejection rate '+r'$N_{B}/N_{B}^{sel}$', fontsize=20)

    ax.set_xticks(list(map(lambda x: round(x,1),np.linspace(0, 1, 21))) )
    ax.set_yticks(np.linspace(0, 1, 11),)


    ax_2.set_yticks(np.linspace(np.amin(bkg_ratios[bkg_ratios!=-1]), np.amax(bkg_ratios[bkg_ratios!=-1]), 11))

    ax_2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_2.yaxis.get_major_formatter().set_scientific(True)

    ax.set_ylim(0, 1.4)
    ax_2.set_ylim(np.amin(bkg_ratios[bkg_ratios!=-1]), 1.4*np.amax(bkg_ratios[bkg_ratios!=-1]))




    plt.subplot(3, 1, 2)
    ax = plt.gca()
    ax_2 = ax.twinx()

    ax.tick_params(labelsize=15)
    ax_2.tick_params(labelsize=15)

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=25, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    tag = 'Simulation @{}GeV'.format(ep) if ep != None else 'Simulation'

    plt.text(0.1, 0.8, tag, fontsize=23, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    l1 = ax.plot(threshold_lists, purities, linewidth=linewidth, color='red',
                 label='{} signal purity'.format(signal_greek))
    l2 = ax_2.plot(threshold_lists[bkg_ratios != -1], bkg_ratios[bkg_ratios != -1], linewidth=linewidth, color='black',
                   label='Background rejection rate')

    lns = l1 + l2
    if -1 in bkg_ratios:
        l3 = ax_2.plot(threshold_lists[bkg_ratios == -1],
                       np.amax(bkg_ratios[bkg_ratios != -1]) * np.ones(np.sum((bkg_ratios == -1) != 0)),
                       '*', color='orange',
                       label='Pure {}'.format(signal_greek))
        lns += l3

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.9, 0.95),fontsize=15)

    ax.set_xlabel('ANN probability cut', fontsize=20)

    ax.set_ylabel('Purity '+r'${N_{S}^{sel}}/({N_{B}^{sel}+N_{S}^{sel}})$', fontsize=20)
    ax_2.set_ylabel('Efficiency '+r'$N_{B}/N_{B}^{sel}$', fontsize=20)

    ax.set_xticks(list(map(lambda x: round(x,1),np.linspace(0, 1, 21))) )
    ax.set_yticks(np.linspace(0, 1, 11), )
    ax_2.set_yticks(np.linspace(np.amin(bkg_ratios[bkg_ratios != -1]), np.amax(bkg_ratios[bkg_ratios != -1]), 11,))

    ax.set_ylim(0, 1.4)
    ax_2.set_ylim(np.amin(bkg_ratios[bkg_ratios != -1]), 1.4 * np.amax(bkg_ratios[bkg_ratios != -1]))






    plt.subplot(3, 1, 3)
    ax = plt.gca()

    ax.tick_params(labelsize=15)

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=25, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    tag = 'Simulation @{}GeV'.format(ep) if ep != None else 'Simulation'

    plt.text(0.1, 0.8, tag, fontsize=23, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    lns=ax.plot(threshold_lists[significances!=-1], significances[significances!=-1], linewidth=linewidth, color='red',
                 label='{} significance'.format(signal_greek))

    if -1 in bkg_ratios:
        l3 = ax.plot(threshold_lists[significances == -1],
                       np.amax(significances[significances != -1]) * np.ones(np.sum((significances == -1) != 0)),
                       '*', color='orange',
                       label='Pure {}'.format(signal_greek))
        lns += l3

    labs = [l.get_label() for l in lns]

    ax.legend(lns,labs,
              loc='upper right', bbox_to_anchor=(0.9, 0.95), fontsize=15)

    ax.set_xlabel('ANN probability cut', fontsize=20)

    ax.set_ylabel('Significance '+r'${N_{S}^{sel}}/\sqrt{N_{B}^{sel}}$', fontsize=20)



    ax.set_xticks(list(map(lambda x: round(x,1),np.linspace(0, 1, 21))) )

    ax.set_yticks(np.linspace(np.amin(significances[significances != -1]), np.amax(significances[significances != -1]), 11), )


    ax.set_ylim(np.amin(significances[significances != -1]), 1.4 * np.amax(significances[significances != -1]))


    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_merged_evaluation(raw_root_label_dir, ann_root_file_dir,ep_list, label_dict, signal_label, threshold_lists,save_path):

    labels_merged=[]
    ann_scores_merged=[]

    for ep in ep_list:
        dir=str(ep)+'GeV'

        raw_label_path_=os.path.join(raw_root_label_dir, dir+'/labels.npy')
        ann_path_ = os.path.join(ann_root_file_dir, dir + '/imgs_ANN.root')


        labels_=np.load(raw_label_path_)
        ann_scores_=read_ann_score(file_pid_path=ann_path_)

        assert len(labels_) == len(ann_scores_)

        labels_merged.append(labels_)
        ann_scores_merged.append(ann_scores_)

    plot_evaluation(
        raw_label_path=None,
        ann_file_path=None,
        save_path=save_path,
        signal_label=signal_label,
        label_dict=label_dict,
        threshold_lists=threshold_lists,
        raw_labels=np.concatenate(labels_merged,axis=0),
        ann_scores=np.concatenate(ann_scores_merged,axis=0),

    )



def plot_merged_ann_score(raw_root_label_dir, ann_root_file_dir, label_dict, stacked, log, bins, ep_list, source, save_dir, n_classes):

    labels_merged=[]
    ann_scores_merged=[]
    for ep in ep_list:
        dir=str(ep)+'GeV'

        raw_label_path_=os.path.join(raw_root_label_dir, dir+'/labels.npy')
        ann_path_ = os.path.join(ann_root_file_dir, dir + '/imgs_ANN.root')


        labels_=np.load(raw_label_path_)
        ann_scores_=read_ann_score(file_pid_path=ann_path_)

        assert len(labels_) == len(ann_scores_)

        labels_merged.append(labels_)
        ann_scores_merged.append(ann_scores_)


    plot_ann_score(raw_label_path=None, ann_file_path=None, save_dir=save_dir, bins=bins, ep=None,
                   raw_labels=np.concatenate(labels_merged,axis=0),
                   ann_scores=np.concatenate(ann_scores_merged,axis=0),
                   stacked=stacked,
                   log=log,label_dict=label_dict,
                   source=source,
                   n_classes=n_classes)

def plot_ann_score(raw_label_path, ann_file_path, save_dir,stacked,log, label_dict, source,  n_classes,bins=100, ep=None,**kwargs):

    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'data': 'Data test set\nData training approach'
    }

    color_dict={
        'muon': 'green',
        'muon_minus': 'green',
        'muon_plus': 'green',
        'electron':'blue',
        'electron_minus': 'blue',
        'positron': 'blue',
        'pion': 'red',
        'pion_minus':'red',
        'pion_plus': 'red',
        'noise':'orange',

    }

    greek_letter_dict={
        'muon': r'Muon',
        'muon_minus': r'$'+chr(956)+'^{-}$',
        'muon_plus': r'$'+chr(956)+'^{+}$',
        'electron': r'Electron',
        'positron': r'$e^{+}$',
        'pion': r'Pion',
        'pion_minus': r'$'+chr(960)+'^{-}$',
        'pion_plus': r'$'+chr(960)+'^{+}$',
        'noise': 'Noise',
    }


    raw_labels=np.load(raw_label_path) if raw_label_path!=None else kwargs.get('raw_labels')

    ann_score=read_ann_score(file_pid_path=ann_file_path, n_classes=n_classes) if ann_file_path !=None else kwargs.get('ann_scores')



    legend_labels=[greek_letter_dict.get(type) for type in label_dict.values()]
    colors = [color_dict.get(type) for type in label_dict.values()]

    label_size=18

    for i , type in label_dict.items():

        ann_scores_to_plot=[ann_score[:,i][raw_labels==label] for label in label_dict.keys()]

        weights=[np.ones(len(scores))/len(raw_labels) for scores in ann_scores_to_plot]

        fig=plt.figure(figsize=(8,7))
        ax = plt.gca()

        plt.text(0.05, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        tag= text_dict.get(source) +' @{}GeV'.format(ep) if ep !=None else text_dict.get(source)

        plt.text(0.05, 0.89, tag, fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        n_count=[]
        hatch_list=['//','//', '\\\\']
        for score, label,color, weight, hatch  in zip(ann_scores_to_plot, legend_labels, colors, weights, hatch_list):
            n,_,_=plt.hist(score, bins=bins, label=label, range=[0, 1], histtype='step',
                     color=color,
                     density=False, stacked=stacked, alpha=1, log=log, linewidth=3,
                     weights=weight,
                     hatch=hatch,
                     )
            n_count.append(n)





        plt.legend(loc='upper right', bbox_to_anchor=(0.92,0.98), fontsize=label_size)

        plt.xticks(np.linspace(0,1,11), fontsize=label_size)
        plt.yticks(fontsize=label_size)
        plt.xlabel('ANN {} likelihood'.format(greek_letter_dict.get(type).lower()), fontsize=label_size)
        plt.ylabel('# [Normalized]'.format(greek_letter_dict.get(type)), fontsize=label_size)

        if 'y_ll_list' in kwargs.keys():
            ax.set_ylim(bottom=kwargs.get('y_ll_list')[i])
        plt.ylim(top=20*np.amax(np.concatenate(n_count)))
        plt.savefig(os.path.join(save_dir, 'ann_score_{}_{}'.format(greek_letter_dict.get(type),source)))
        plt.show()
        plt.close(fig)


def plot_mc_tot_e_purifiled_compare(raw_file_dir, ann_root_dir, save_path, ep_list, threshold,stacked,bins=100):


    label_dict={
        0:'muon_plus',
        1:'positron',
        2:'pion_plus',

    }

    color_dict={
        'muon_minus': 'green',
        'muon_plus': 'green',
        'electron':'blue',
        'positron': 'blue',
        'pion_minus':'red',
        'pion_plus': 'red',
        'noise':'orange',

    }

    greek_letter_dict={
        'muon_minus': r'$'+chr(956)+'^{-}$',
        'muon_plus': r'$'+chr(956)+'^{+}$',
        'electron': r'$e^{-}$',
        'positron': r'$e^{+}$',
        'pion_minus': r'$'+chr(960)+'^{-}$',
        'pion_plus': r'$'+chr(960)+'^{+}$',
        'noise': 'Noise',
    }

    fig = plt.figure(figsize=(24, 15))

    for i, ep in enumerate(ep_list):

        raw_dir_ = os.path.join(raw_file_dir, str(ep) + 'GeV')
        raw_imgs_path=os.path.join(raw_dir_, 'imgs.npy')
        raw_labels_path = os.path.join(raw_dir_, 'labels.npy')

        ann_dir_=os.path.join(ann_root_dir, str(ep) + 'GeV')
        ann_tags_path=os.path.join(ann_dir_,'imgs_ANN.root')

        ann_results=pick_npy_tot_e(file_pid_path=ann_tags_path, imgs_path=raw_imgs_path,threshold=threshold)

        raw_ed_ = np.sum(np.load(raw_imgs_path),axis=(1,2,3))

        raw_labels=np.load(raw_labels_path)


        up_limit = np.mean(raw_ed_) + 2.5 * np.std(raw_ed_)
        # low_limit=np.mean(puri_ed_) - 4 * np.std(puri_ed_)


        truth_eds, truth_labels, pred_labels, colors=[], [], [], []


        for key, type in label_dict.items():


            truth_eds.append((raw_ed_[raw_labels==key]).reshape(-1))
            truth_labels.append(greek_letter_dict.get(type)+' truth')
            pred_labels.append('Predicted '+greek_letter_dict.get(type))
            colors.append(color_dict.get(type))



        plt.subplot(3, 4, i + 1)
        ax = plt.gca()

        plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.text(0.1, 0.8, 'Simulation @{}GeV'.format(ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.text(0.1, 0.7, 'ANN threshold @{}'.format(threshold), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.hist(raw_ed_, bins=bins, label='Mixed beam', range=[0, up_limit], histtype='step',
                 color='black',
                 density=False, linewidth=1.5)



        plt.hist(truth_eds, bins=bins, label=truth_labels, range=[0, up_limit],
                 histtype='step', color=colors,
                 density=False,stacked=stacked, linewidth=1.5)

        plt.hist(ann_results[:len(label_dict.keys())], bins=bins, label=pred_labels, range=[0, up_limit],
                 histtype='stepfilled', color=colors,
                 density=False, stacked=stacked, alpha=0.5)

        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99))
        plt.xlabel('[MeV]')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def crystal_ball_func(x, alpha, n, mean, sigma, ):
    A = ((n / abs(alpha)) ** n) * math.exp(-1 * ((alpha ** 2) / 2))
    B = n / abs(alpha) - abs(alpha)
    C = n / abs(alpha) / (n - 1) * math.exp(-1 * (alpha ** 2) / 2)
    D = math.sqrt(math.pi / 2) * (1 + math.erf(abs(alpha) / math.sqrt(2)))
    N = 1 / (sigma * (C + D))

    return np.where((x - mean) / sigma > (-1 * alpha), N * np.exp(-1 * (np.square(x - mean)) / (2 * sigma ** 2)),
                    N * A * np.power((B - (x - mean) / sigma), (-1 * n)))


def resolution_func(x, a, b):
    return np.sqrt((a / np.sqrt(x))**2 + b**2)

def resolution_noise_term_func(x, a, b, c):
    return np.sqrt((a / np.sqrt(x))**2 + b**2 + (c / x)**2)


def linearity_func(x, a, b):
    return a * x + b


def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def get_cbf_para(file_dir, bins, func, ep_list):
    para_dict = dict()

    for i, ep in enumerate(ep_list):
        file_path = os.path.join(file_dir, str(ep) + 'GeV.npy')

        ed = np.load(file_path)
        up_limit = np.mean(ed) + 3 * np.std(ed)
        num=len(ed[np.logical_and(ed>=0, ed<=up_limit)])


        n, b = np.histogram(ed, bins=bins, range=[0, up_limit], density=True)
        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]
        # alpha=1. n=1.1
        popt, pcov = curve_fit(func, b, n, p0=[0.85, 4.2, b[np.argmax(n)], 10])
        para_dict[ep] = [popt, num]

    return para_dict


def get_guass_para(file_dir, bins, func, low_n_sigma_lists, up_n_sigma_lists, ep_list):
    para_dict = dict()


    for i, ep in enumerate(ep_list):
        file_path = os.path.join(file_dir, str(ep) + 'GeV.npy')

        ed = np.load(file_path)
        std = np.std(ed)
        mean = np.mean(ed)

        n, b = np.histogram(ed, bins=bins, range=[0, mean + 3 * std], density=True)
        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        up_limit = mean + up_n_sigma_lists[i] * std
        lower_limit = mean - low_n_sigma_lists[i] * std
        b2 = b[np.logical_and(b > lower_limit, b < up_limit)]
        n2 = n[np.logical_and(b > lower_limit, b < up_limit)]

        num_=len(ed[np.logical_and(ed>=lower_limit,ed<=up_limit)])

        popt, pcov = curve_fit(func, b2, n2, p0=[1, 1.1 * b[np.argmax(n)], 10])
        para_dict[ep] = [popt,num_]

    return para_dict


def plot_main(root_dir,ep_list ):


    plot_tot_e(file_dir=root_dir,
               save_path=os.path.join(root_dir, 'picked_pi.png'),
               func=crystal_ball_func, bins=500, ep_list=ep_list)

    para_dict = get_cbf_para(file_dir=root_dir,
                             bins=500,
                             func=crystal_ball_func,
                             ep_list=ep_list)

    plot_resolution(para_dict, resolution_func, ep_list=ep_list, save_path=os.path.join(root_dir, 'picked_pi_resolution.png'))
    # plot_resolution(para_dict, resolution_noise_term_func, ep_list=ep_list, save_path=os.path.join(root_dir, 'picked_pi_resolution_nt.png'))
    plot_linearity(para_dict, linearity_func, ep_list=ep_list, save_path=os.path.join(root_dir, 'picked_pi_linearity.png'))

    low_n_sigma_lists = [1.4, 1.3, 1, 0.8,
                         0.5, 0.3, 0.2, 0.15,
                         0.15, 0.15, 0.15]
    up_n_sigma_lists = [1.5, 1.5, 1.3, 1.3,
                        1, 1, 1, 1,
                        1, 0.9, 0.9]
    plot_tot_e_guass(file_dir=root_dir,
                     save_path=os.path.join(root_dir, 'picked_pi_gauss.png'),
                     func=gaussian_func, bins=500, low_n_sigma_lists=low_n_sigma_lists,
                     up_n_sigma_lists=up_n_sigma_lists,ep_list=ep_list)

    para_dict = get_guass_para(
        file_dir=root_dir,
        bins=500,
        func=gaussian_func, low_n_sigma_lists=low_n_sigma_lists, up_n_sigma_lists=up_n_sigma_lists,ep_list=ep_list)

    plot_resolution(para_dict, resolution_func,ep_list=ep_list,
                    save_path=os.path.join(root_dir, 'picked_pi_resolution_gauss.png'))
    plot_resolution(para_dict, resolution_noise_term_func, ep_list=ep_list,
                    save_path=os.path.join(root_dir, 'picked_pi_resolution_nt_gauss.png'))
    plot_linearity(para_dict, linearity_func,ep_list=ep_list,
                   save_path=os.path.join(root_dir, 'picked_pi_linearity_gauss.png'))

def plot_selection_efficiency(ann_effi_path, particle_label, fd_effi, save_path):

    ann_effi_file = pd.read_csv(ann_effi_path)
    ep=list(ann_effi_file.columns)
    ep=list(map(lambda x: x.replace('GeV', ''), ep))


    ann_effi=ann_effi_file.values[particle_label].reshape(-1)

    fig=plt.figure(figsize=(6,5))
    ax=plt.gca()

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.83, 'Beam data', fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    if fd_effi==None:
        bar=plt.bar(ep, list(map(lambda x: round(x*100,1), ann_effi)),
                label='ANN selection rate',
                width=0.5,
                alpha=0.8
               )
        plt.bar_label(bar, label_type='edge')

    plt.ylim(0,130)
    plt.yticks(np.linspace(0,100, 11))
    plt.xlabel('Energy [GeV]')
    plt.ylabel('[%]')

    plt.legend(loc='upper left', bbox_to_anchor=(0.1,0.8))

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_fd_hit_e(file_dir, save_path, ep_list, bins=500 ):
    fig = plt.figure(figsize=(24, 15))

    for i, ep in enumerate(ep_list):
        signal_file_path = os.path.join(file_dir, str(ep) + 'GeV_fd_signal.npy')
        bkg_file_path = os.path.join(file_dir, str(ep) + 'GeV_fd_bkg.npy')

        signal = np.load(signal_file_path)
        bkg= np.load(bkg_file_path)

        plt.subplot(3, 4, i + 1)
        ax = plt.gca()

        plt.text(0.05, 0.9, 'CEPC Test Beam', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )
        plt.text(0.05, 0.8, '{} data @{}GeV'.format(chr(960), ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        # cmap1 = LinearSegmentedColormap.from_list('custom_cmap', ['red', 'red'], N=256)

        plt.hist2d(signal[0], signal[1], bins=bins, density=False, cmap='Reds', label=chr(960), range=[[0,10], [0, 1.5]],
                   cmin=0.1,norm=LogNorm() )

        # cmap2 = LinearSegmentedColormap.from_list('custom_cmap', ['black', 'black'], N=256)

        # plt.hist2d(bkg[0], bkg[1], bins=bins, density=False, cmap='Greys', label='Backgrounds', range=[[0,10], [0, 1.5]],
        #            alpha=0.5,cmin=0.1,norm=LogNorm() )

        plt.colorbar()

        plt.ylabel(r'$FD_{2D}$')
        plt.xlabel(r'$<E_{Hit}>$')

        plt.xlim(0,10)
        plt.ylim(0,1.5)

    # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.75))

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_composition(file_path, save_path, source, threshold):

    label_dict={
        0:'Muon',
        1:'Electron',
        2:'Pion',
        3:'Noise'
    }
    file=pd.read_csv(file_path)
    ep = list(file.columns)
    ep = list(map(lambda x: int(x.replace('GeV', '')), ep))

    compositions = file.values

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.83, source, fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.76, 'ANN threshold @{}'.format(threshold), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    bar_width=3
    x_base=np.linspace(0, 17*(len(ep)-1), len(ep))

    for key, value in label_dict.items():
        plt.bar(x_base+(key-1.5)*bar_width, list(map(lambda x: round(x * 100, 2), compositions[key])),
                      label=value,
                      width=bar_width,
                      alpha=0.8
                          )

    plt.xticks(x_base,ep)
    plt.ylim(0, 120)
    plt.yticks(np.linspace(0, 100, 11))
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Classified particle composition [%]')

    plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.98))

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)





if __name__ == '__main__':
    # TODO =================================== check ==================================================
    # threshold = 0
    # ckp_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.001_batch32_optim_SGD_classes_4_ihep_v1'
    # root_dir = os.path.join(ckp_dir, 'ANA/ann_picked_pi_plus')
    # ep_list = [10, 20, 30, 40, 50, 60, 70, 80]
    # TODO =================================== check ==================================================
    #
    # plot_main(root_dir=root_dir,
    #           ep_list=ep_list)

    # puri_file_dir_dict = {
    #     'pion_plus': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_pi_plus_{}'.format(int(100*threshold)),
    #     'muon_plus': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_mu_plus_{}'.format(int(100*threshold)),
    #     'positron': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_e_plus_{}'.format(int(100*threshold)),
    #     'noise': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_noise_2022_{}'.format(int(100*threshold)),
    # }
    #
    # plot_tot_e_purifiled_compare(raw_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2022/v1',
    #                              puri_file_dir_dict=puri_file_dir_dict,
    #                              save_path=os.path.join(ckp_dir, 'ANA/Fig/purifiled_compare_stacked_2022_{}.png'.format(int(100*threshold))),
    #                              ep_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
    #                              threshold=threshold,
    #                              stacked=True
    #                              )
    #



    # plot_selection_efficiency(ann_effi_path=os.path.join(root_dir, 'selection_efficiency.csv'),
    #                           fd_effi=None,
    #                           save_path=os.path.join(ckp_dir,'ANA/Fig/selection_rate_2022.png'))
    #


    # plot_composition(
    #     file_path=os.path.join(root_dir,'composition.csv'),
    #     save_path=os.path.join(root_dir,'composition.png'),
    #     source='2022 SPS-H8',
    #     threshold=0.3
    # )


    # plot_mc_tot_e_purifiled_compare(raw_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0515_version_sep_no_noise',
    #                                 ann_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/mc_0515/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #                                 save_path=os.path.join(ckp_dir,'ANA/Fig/purifiled_compare_mc.png'),
    #                                 ep_list=ep_list,
    #                                 threshold=threshold,
    #                                 stacked=False)

    # plot_composition(
    #     file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/mc_0515/composition.csv',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/mc_0515/composition.png',
    #     source='Simulation',
    #     threshold=0.9
    # )

    # label_dict = {
    #     0: 'muon_plus',
    #     1: 'positron',
    #     2: 'pion_plus',
    # }

    # plot_evaluation(
    #     raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0515_version_sep_no_noise/50GeV/labels.npy',
    #     ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/mc_0515_sep_no_noise/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/50GeV/imgs_ANN.root',
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/mc_0515/eva_50.png',
    #     ep=50,
    #     signal_label=2,
    #     stacked=False,
    #     label_dict=label_dict,
    #     threshold_lists=np.linspace(0,0.999, 100)
    #
    # )

    # for label in label_dict.keys():
    #     plot_merged_evaluation(
    #         raw_root_label_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0515_version_sep_no_noise',
    #         ann_root_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/mc_0515_sep_no_noise/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #         save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/Fig_mc/eva_signal_{}.png'.format(label),
    #         signal_label=label,
    #         label_dict=label_dict,
    #         threshold_lists=np.linspace(0, 0.999, 100),
    #         ep_list=np.arange(10,121,10),
    #
    #     )
    # plot_ann_score(raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0515_version_sep_no_noise/50GeV/labels.npy',
    #                ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/mc_0515/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/50GeV/imgs_ANN.root',
    #                save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/mc_0515/ann_50.png',
    #                stacked=False,
    #                label_dict=label_dict,
    #                log=True
    #                )

    # plot_merged_ann_score(
    #     raw_root_label_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0515_version_sep_no_noise',
    #     ann_root_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/mc_0515_sep_no_noise/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2',
    #     bins=100,
    #     ep_list=np.arange(10,121,10),
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/mc_0515/ann_score_merged_log_stacked.png',
    #     stacked=True,
    #     log=True，label_dict =label_dict
    # )
    # TODO =================================== check ==================================================
    # threshold = 0
    # ckp_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1'
    # root_dir = os.path.join(ckp_dir, 'ANA/ann_picked_pi_minus')
    # ep_list =[10, 15, 20, 30, 40, 50, 60, 70, 80]
    # TODO =================================== check ==================================================
    #
    # plot_main(root_dir=root_dir,
    #           ep_list=ep_list)

    # puri_file_dir_dict = {
    #     'pion_minus': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_pi_minus_{}'.format(int(100*threshold)),
    #     'muon_minus': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_mu_minus_{}'.format(int(100*threshold)),
    #     'electron': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_e_minus_{}'.format(int(100*threshold)),
    #     'noise': '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0607_res_epoch_200_lr_0.001_batch64_optim_SGD_classes_4_ihep_v2/ANA/ann_picked_noise_2023_{}'.format(int(100*threshold)),
    # }
    #
    # plot_tot_e_purifiled_compare(raw_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2023/v4',
    #                              puri_file_dir_dict=puri_file_dir_dict,
    #                              save_path=os.path.join(ckp_dir, 'ANA/Fig/purifiled_compare_stacked_{}.png'.format(int(100*threshold))),
    #                              ep_list=[10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120],
    #                              threshold=threshold,
    #                              stacked=True
    #                              )
    #

    # plot_tot_e_purifiled_compare(raw_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ahcal_alone_2023/v4',
    #                              puri_file_dir=root_dir,
    #                              save_path=os.path.join(ckp_dir,'ANA/Fig/purifiled_compare_2023.png'),
    #                              ep_list=ep_list)

    #
    # plot_selection_efficiency(ann_effi_path=os.path.join(root_dir, 'selection_efficiency.csv'),
    #                           fd_effi=None,
    #                           save_path=os.path.join(ckp_dir,'ANA/Fig/selection_rate_2023.png'))
    #
    #
    # plot_fd_hit_e(file_dir=root_dir,
    #               save_path=os.path.join(ckp_dir,'ANA/Fig/fd_2023.png'),
    #               ep_list=ep_list,
    #               bins=500)
    #
    # plot_composition(
    #     file_path=os.path.join(root_dir, 'composition.csv'),
    #     save_path=os.path.join(root_dir, 'composition.png'),
    #     source='2023 SPS-H2',
    #     threshold=threshold
    # )

    # TODO v1
    label_dict = {
        0: 'muon',
        1: 'electron',
        2: 'pion',
    }
    plot_ann_score(raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720/TV/labels.npy',
                   ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/TV/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/imgs_ANN.root',
                   save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_1e-05_batch32_optim_Adam_classes_4_l_gamma_0.5_step_10v1/ANA/Fig',
                   label_dict=label_dict,
                   stacked=False,
                   bins=20,
                   log=True,
                   source='data',
                   y_ll_list=[None, 5e-5, None],
                   n_classes=4
                   )

    label_dict = {
        0: 'muon',
        1: 'electron',
        2: 'pion',
    }
    plot_ann_score(raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version_block_1_1/TV/labels.npy',
                   ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/PIDTags/TV/imgs_ANN.root',
                   save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0901_mc_res18_avg_epoch_200_lr_0.001_batch_64_optim_SGD_classes_4_l_gamma_0.5_step_100_st_True_b_1_1_v1/ANA/Fig',
                   label_dict=label_dict,
                   stacked=False,
                   bins=30,
                   log=True,
                   source='mc',
                   y_ll_list=[1e-5, None, None],
                   n_classes=4)

    # TODO v2
    # label_dict = {
    #     0: 'muon',
    #     1: 'electron',
    #     2: 'pion',
    # }
    # plot_ann_score(raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720_no_noise/Test/labels.npy',
    #                ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_tb/Test/0728_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_3_v1/imgs_ANN.root',
    #                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0728_tb_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_3_v1/ANA/Fig',
    #                label_dict=label_dict,
    #                stacked=False,
    #                bins=100,
    #                log=True,
    #                source='data',
    #                y_ll_list=[None, None, None],
    #                n_classes=3)

    # label_dict = {
    #     0: 'muon',
    #     1: 'electron',
    #     2: 'pion',
    # }
    # plot_ann_score(raw_label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version/TV/labels.npy',
    #                ann_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/0720_mc/TV/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/imgs_ANN.root',
    #                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0720_mc_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_v1/ANA/Fig_mc',
    #                label_dict=label_dict,
    #                stacked=False,
    #                bins=20,
    #                log=True,
    #                source='mc',
    #                y_ll_list=[1e-5, None, None])
    # root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1/ANA/2022_pi_beam_info'
    # plot_composition(
    #     file_path=os.path.join(root_dir, 'composition.csv'),
    #     save_path=os.path.join(root_dir, 'composition.png'),
    #     source='2023 SPS-H8',
    #     threshold=0.92,
    # )
    root_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1/ANA/2023_pi_beam_info_ps'
    plot_composition(
        file_path=os.path.join(root_dir, 'composition.csv'),
        save_path=os.path.join(root_dir, 'composition.png'),
        source='2023 PS-T9',
        threshold=0.95
    )
    pass
