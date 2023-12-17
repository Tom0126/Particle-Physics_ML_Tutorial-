#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 14:21
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : bdt_var_compare.py
# @Software: PyCharm
import glob

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import ks_2samp, chisquare

class Shower_Compare():

    def __init__(self,mc_var_path, e_beam_var_path, pi_beam_var_path, save_dir, ep):

        self.mc_var_path=mc_var_path
        self.e_var_path=e_beam_var_path
        self.pi_var_path=pi_beam_var_path

        self.save_dir=save_dir

        self.mc_var = None
        self.beam_var = None
        self.ep=ep



    def load(self):

        self.mc_var=pd.read_csv(self.mc_var_path)
        self.e_var=pd.read_csv(self.e_var_path)
        self.pi_var=pd.read_csv(self.pi_var_path)

        self.mc_label = self.mc_var['Particle_label']
        self.e_label = self.e_var['Particle_label']
        self.pi_label = self.pi_var['Particle_label']

    def compare(self, column, bins, e_scale=1, pi_scale=1):

        mc_column = self.mc_var[column].values
        e_column = self.e_var[column].values
        pi_column = self.pi_var[column].values



        range=[np.amin(np.concatenate([mc_column[np.logical_and(self.mc_label >= 0, self.mc_label <= 2)],
                                       e_column[np.logical_and(self.e_label >= 0, self.e_label <= 2)],
                                       pi_column[np.logical_and(self.pi_label >= 0, self.pi_label <= 2)]])),
               np.amax(np.concatenate([mc_column[np.logical_and(self.mc_label >= 0, self.mc_label <= 2)],
                                       e_column[np.logical_and(self.e_label >= 0, self.e_label <= 2)],
                                       pi_column[np.logical_and(self.pi_label >= 0, self.pi_label <= 2)]]))]


        fig = plt.figure(figsize=(6, 5))

        ax = plt.gca()

        plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.text(0.1, 0.8, 'Incident particle @{}GeV'.format(self.ep), fontsize=12, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.hist(mc_column[self.mc_label == 0], bins=bins, histtype='step', label='MC mu', color='green',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 0]))/len(mc_column[self.mc_label == 0]),
                 linewidth=2)

        plt.hist(mc_column[self.mc_label == 1], bins=bins, histtype='step', label='MC e',
                 color='blue',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 1]))/len(mc_column[self.mc_label == 1]),
                 linewidth=2)

        plt.hist(e_column[self.e_label == 1]*e_scale, bins=bins, histtype='stepfilled', alpha=0.5,
                 label='Data e',
                 color='blue', range=range,
                 weights=np.ones(len(e_column[self.e_label == 1]))/len(e_column[self.e_label == 1]),)

        plt.hist(mc_column[self.mc_label == 2], bins=bins, histtype='step', label='MC pi', color='red',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 2]))/len(mc_column[self.mc_label == 2]),
                 linewidth=2)

        plt.hist(pi_column[self.pi_label == 2]*pi_scale, bins=bins, histtype='stepfilled', alpha=0.5,
                 label='Data pi',
                 color='red', range=range,
                 weights=np.ones(len(pi_column[self.pi_label == 2]))/len(pi_column[self.pi_label == 2]))

        plt.legend(loc='upper right', bbox_to_anchor=(0.9,0.95))
        plt.xlabel(column.replace('_', ' '),fontsize=13)
        plt.ylabel('# [Normalized]', fontsize=13)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, column + '.png'))
        # plt.show()
        plt.close(fig)

class Full_Shower_Compare(Shower_Compare):

    def __init__(self,mc_var_path, mu_beam_var_path, e_beam_var_path, pi_beam_var_path, save_dir, ep):

        super().__init__(mc_var_path, e_beam_var_path, pi_beam_var_path, save_dir, ep)
        self.mu_var_path=mu_beam_var_path

    def load(self):

        self.mc_var=pd.read_csv(self.mc_var_path)
        self.mu_var = pd.read_csv(self.mu_var_path)
        self.e_var=pd.read_csv(self.e_var_path)
        self.pi_var=pd.read_csv(self.pi_var_path)
        self.mc_label = self.mc_var['Particle_label']
        self.mu_label = self.mu_var['Particle_label']
        self.e_label = self.e_var['Particle_label']
        self.pi_label = self.pi_var['Particle_label']

    def compare(self, column, bins, e_scale=1, pi_scale=1, ll=None, ul=None, log=False, y_ul=None, l_x=0.9, x_label=None):


        mc_column = self.mc_var[column].values
        mu_column=self.mu_var[column].values
        e_column = self.e_var[column].values
        pi_column = self.pi_var[column].values

        if column == 'Shower_radius':
            cut = mu_column < 1.2
            mu_column = mu_column[cut]
            self.mu_label = self.mu_label[cut]

        if column == 'Shower_layer_ratio':
            cut = mu_column < 0.1
            mu_column = mu_column[cut]
            self.mu_label = self.mu_label[cut]

        # if column == 'FD_6':
        #     cut = mu_column < 1.23
        #     mu_column = mu_column[cut]
        #     self.mu_label = self.mu_label[cut]

        if column == 'Shower_layer':
            cut = mu_column < 5
            mu_column = mu_column[cut]
            self.mu_label = self.mu_label[cut]


        if ll==None:
            ll = np.amin(np.concatenate([mc_column[np.logical_and(self.mc_label>=0, self.mc_label<=2)],
                                       mu_column[np.logical_and(self.mu_label>=0, self.mu_label<=2)],
                                       e_column[np.logical_and(self.e_label>=0, self.e_label<=2)],
                                       pi_column[np.logical_and(self.pi_label>=0, self.pi_label<=2)]]))

        if ul==None:

            ul= np.amax(np.concatenate([mc_column[np.logical_and(self.mc_label>=0, self.mc_label<=2)],
                                       mu_column[np.logical_and(self.mu_label>=0, self.mu_label<=2)],
                                       e_column[np.logical_and(self.e_label>=0, self.e_label<=2)],
                                       pi_column[np.logical_and(self.pi_label>=0, self.pi_label<=2)]]))

        range=[ll,ul]


        fig = plt.figure(figsize=(8, 7))

        ax = plt.gca()
        label_size=21
        plt.text(0.05, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.05, 0.89, 'BDT input', fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        # if self.ep==None:
        #     plt.text(0.15, 0.85, 'Variable comparison', fontsize=label_size, fontstyle='normal',
        #              horizontalalignment='left',
        #              verticalalignment='center', transform=ax.transAxes, )
        if self.ep!=None:

            plt.text(0.15, 0.89, 'Energy @{}GeV'.format(self.ep), fontsize=label_size, fontstyle='normal',
                     horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes, )

        # plt.hist(mc_column[self.mc_label == 0], bins=bins, histtype='step', label='Muon MC', color='green',
        #          range=range,
        #          weights=np.ones(len(mc_column[self.mc_label == 0]))/len(mc_column[self.mc_label == 0]),
        #          linewidth=3,
        #          alpha=1,
        #          log=log,
        #          hatch='/')
        #
        # plt.hist(mu_column[self.mu_label == 0] * e_scale, bins=bins, histtype='step',  alpha=1,
        #          # label='Muon Data',
        #          color='green', range=range, linewidth=2,
        #          weights=np.ones(len(mu_column[self.mu_label == 0])) / len(mu_column[self.mu_label == 0]), log=log )

        # n_mu, b_mu = np.histogram(mu_column[self.mu_label == 0] * e_scale, bins=bins, range=[ll, ul],
        #                     weights=np.ones(len(mu_column[self.mu_label == 0])) / len(mu_column[self.mu_label == 0]))
        #
        # plt.plot((b_mu[:-1] + 0.5 * (b_mu[1:] - b_mu[:-1]))[n_mu>0], n_mu[n_mu>0], 'o', color='green', alpha=1, label='Muon Data', markersize=7)

        plt.hist(mc_column[self.mc_label == 0], bins=bins, histtype='step',label='Electron MC',
                 color='blue',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 0]))/len(mc_column[self.mc_label == 0]),
                 linewidth=3,
                 alpha=1,
                 log=log,
                 hatch='\\')




        # plt.hist(e_column[self.e_label == 1]*e_scale, bins=bins, histtype='step', alpha=1,
        #          # label='Electron Data',
        #          color='blue', range=range, linewidth=2,
        #          weights=np.ones(len(e_column[self.e_label == 1]))/len(e_column[self.e_label == 1]),log=log)
        #
        # n_e, b_e = np.histogram(e_column[self.e_label == 1]*e_scale, bins=bins, range=[ll, ul],
        #                           weights=np.ones(len(e_column[self.e_label == 1]))/len(e_column[self.e_label == 1]))
        #
        # plt.plot((b_e[:-1] + 0.5 * (b_e[1:] - b_e[:-1]))[n_e>0], n_e[n_e>0], 'o', color='blue', alpha=1, label='Electron Data',
        #          markersize=7)

        plt.hist(mc_column[self.mc_label == 1], bins=bins, histtype='step', label='Pion MC', color='red',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 1]))/len(mc_column[self.mc_label == 1]),
                 linewidth=3,
                 alpha=1,
                 log=log,
                 hatch='//')
        #
        # plt.hist(pi_column[self.pi_label == 2]*pi_scale, bins=bins,histtype='step', alpha=1,
        #          # label='Pion Data',
        #          color='red', range=range, linewidth=2,
        #          weights=np.ones(len(pi_column[self.pi_label == 2]))/len(pi_column[self.pi_label == 2]),
        #          log=log)
        #
        # n_pi, b_pi = np.histogram(pi_column[self.pi_label == 2]*pi_scale, bins=bins, range=[ll, ul],
        #                         weights=np.ones(len(pi_column[self.pi_label == 2]))/len(pi_column[self.pi_label == 2]))
        #
        # plt.plot((b_pi[:-1] + 0.5 * (b_pi[1:] - b_pi[:-1]))[n_pi>0], n_pi[n_pi>0], 'o', color='red', alpha=1, label='Pion Data',
        #          markersize=7)

        plt.tick_params(labelsize=label_size-2)
        plt.legend(bbox_to_anchor=(0.95, 0.98), bbox_transform=ax.transAxes, loc='upper right', fontsize=label_size)
        x_label=x_label if x_label !=None else column.replace('_', ' ').title()

        plt.xlabel(x_label,fontsize=label_size+2)
        plt.ylabel('# [Normalized]', fontsize=label_size)
        if y_ul!=None:
            plt.ylim(top=y_ul)
            plt.yticks(np.round(np.linspace(0,y_ul,7),decimals=2))
        if ll!=None:
            plt.xlim(left=ll)
        if ul!=None:
            plt.xlim(right=ul)
        if ll!=None and ul!=None:
            plt.xticks(np.round(np.linspace(ll,ul,7),decimals=2))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, column + '.png'))
        plt.show()
        plt.close(fig)


    def ks_test(self, column,):

        mc_column = self.mc_var[column].values
        mu_column = self.mu_var[column].values
        e_column = self.e_var[column].values
        pi_column = self.pi_var[column].values

        print('======================{} {}GeV====================='.format(column,self.ep))
        print('muon')
        print(ks_2samp(mc_column[self.mc_label==0], mu_column[self.mu_label==0]))
        print('e')
        print(ks_2samp(mc_column[self.mc_label == 1], e_column[self.e_label == 1]))
        print('pion'.format(column, self.ep))
        print(ks_2samp(mc_column[self.mc_label == 2], pi_column[self.pi_label == 2]))
        print('===================================================\n')

    def chi_test(self, column, bins):

        mc_column = self.mc_var[column].values
        mu_column = self.mu_var[column].values
        e_column = self.e_var[column].values
        pi_column = self.pi_var[column].values




        mu_mc=mc_column[self.mc_label==0]
        mu_data=mu_column[self.mu_label==0]
        mu_range=[np.amin(mu_data), np.amax(mu_data)]

        f_mu_mc,_= np.histogram(mu_mc,bins=bins, range=mu_range)
        f_mu_data, _= np.histogram(mu_data,bins=bins, range=mu_range)

        f_mu_mc=f_mu_mc[f_mu_data!=0]
        f_mu_data=f_mu_data[f_mu_data!=0]

        e_mc = mc_column[self.mc_label == 1]
        e_data = e_column[self.e_label == 1]
        e_range = [np.amin(e_data), np.amax(e_data)]

        f_e_mc, _ = np.histogram(e_mc, bins=bins, range=e_range)
        f_e_data, _ = np.histogram(e_data, bins=bins, range=e_range)

        f_e_mc=f_e_mc[f_e_data!=0]
        f_e_data=f_e_data[f_e_data!=0]

        pi_mc = mc_column[self.mc_label == 2]
        pi_data = pi_column[self.pi_label == 2]
        pi_range = [np.amin(pi_data), np.amax(pi_data)]

        f_pi_mc, _ = np.histogram(pi_mc, bins=bins, range=pi_range)
        f_pi_data, _ = np.histogram(pi_data, bins=bins, range=pi_range)

        f_pi_mc=f_pi_mc[f_pi_data!=0]
        f_pi_data=f_pi_data[f_pi_data!=0]



        print('======================{} {}GeV====================='.format(column,self.ep))
        print('muon')
        print(chisquare(f_mu_mc/np.sum(f_mu_mc),f_exp= f_mu_data/np.sum(f_mu_data)))
        print('e')
        print(chisquare(f_e_mc/np.sum(f_e_mc), f_exp=f_e_data/np.sum(f_e_data)))
        print('pion'.format(column, self.ep))
        print(chisquare(f_pi_mc/np.sum(f_pi_mc), f_exp=f_pi_data/np.sum(f_pi_data)))
        print('===================================================\n')

class Beam_DATA_COMPARE(Full_Shower_Compare):

    def __init__(self, mc_var_path, mu_beam_var_path, e_beam_var_path, pi_beam_var_path, save_dir, ep, source):
        super().__init__(mc_var_path, mu_beam_var_path, e_beam_var_path, pi_beam_var_path, save_dir, ep)

        self.source=source

    def compare(self, column, bins, e_scale=1, pi_scale=1, ll=None, ul=None, log=False, y_ul=None, l_x=0.9,
                    x_label=None):

        self.load()
        text_dict = {
            'mc': 'MC training approach',
            'tb': 'Data training approach'
        }

        mc_column = self.mc_var[column].values
        mu_column=self.mu_var[column].values
        e_column = self.e_var[column].values
        pi_column = self.pi_var[column].values

        if column=='Shower_radius':
            cut=mu_column<1.2
            mu_column=mu_column[cut]
            self.mu_label=self.mu_label[cut]

        if column == 'Shower_layer_ratio':
            cut = mu_column < 0.1
            mu_column = mu_column[cut]
            self.mu_label = self.mu_label[cut]

        if ll==None:
            ll = np.amin(np.concatenate([mc_column[np.logical_and(self.mc_label>=0, self.mc_label<=2)],
                                       mu_column[np.logical_and(self.mu_label>=0, self.mu_label<=2)],
                                       e_column[np.logical_and(self.e_label>=0, self.e_label<=2)],
                                       pi_column[np.logical_and(self.pi_label>=0, self.pi_label<=2)]]))

        if ul==None:

            ul= np.amax(np.concatenate([mc_column[np.logical_and(self.mc_label>=0, self.mc_label<=2)],
                                       mu_column[np.logical_and(self.mu_label>=0, self.mu_label<=2)],
                                       e_column[np.logical_and(self.e_label>=0, self.e_label<=2)],
                                       pi_column[np.logical_and(self.pi_label>=0, self.pi_label<=2)]]))

        range=[ll,ul]


        fig = plt.figure(figsize=(8, 7))

        ax = plt.gca()
        label_size=15
        plt.text(0.15, 0.9, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )

        # if self.ep==None:
        #     plt.text(0.15, 0.85, 'Variable comparison', fontsize=label_size, fontstyle='normal',
        #              horizontalalignment='left',
        #              verticalalignment='center', transform=ax.transAxes, )
        if self.ep!=None:

            plt.text(0.15, 0.82, 'Incident particle @{}GeV'.format(self.ep), fontsize=label_size, fontstyle='normal',
                     horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes, )

        else:
            plt.text(0.15, 0.82, text_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                     horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes, )
        print(text_dict.get(self.source))
        plt.hist(mc_column[self.mc_label == 0], bins=bins, histtype='stepfilled', linestyle='dashed', label='Muon MC', color='green',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 0]))/len(mc_column[self.mc_label == 0]),
                 linewidth=3,
                 alpha=0.5,
                 log=log)
        #
        plt.hist(mu_column[self.mu_label == 0] * e_scale, bins=bins, histtype='step',  alpha=1,
                 # label='Muon Data',
                 color='green', range=range, linewidth=2,
                 weights=np.ones(len(mu_column[self.mu_label == 0])) / len(mu_column[self.mu_label == 0]), log=log )

        n_mu, b_mu = np.histogram(mu_column[self.mu_label == 0] * e_scale, bins=bins, range=[ll, ul],
                                  weights=np.ones(len(mu_column[self.mu_label == 0])) / len(
                                      mu_column[self.mu_label == 0]))

        plt.plot((b_mu[:-1] + 0.5 * (b_mu[1:] - b_mu[:-1]))[n_mu > 0], n_mu[n_mu > 0], 'o', color='green', alpha=1,
                 label='Muon Data', markersize=7)

        plt.hist(mc_column[self.mc_label == 1], bins=bins, histtype='stepfilled', linestyle='dashed',label='Electron MC',
                 color='blue',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 1]))/len(mc_column[self.mc_label == 1]),
                 linewidth=3,
                 alpha=0.5,
                 log=log)


        plt.hist(e_column[self.e_label == 1]*e_scale, bins=bins, histtype='step', alpha=1,
                 # label='Electron Data',
                 color='blue', range=range, linewidth=2,
                 weights=np.ones(len(e_column[self.e_label == 1]))/len(e_column[self.e_label == 1]),log=log)

        n_e, b_e = np.histogram(e_column[self.e_label == 1] * e_scale, bins=bins, range=[ll, ul],
                                weights=np.ones(len(e_column[self.e_label == 1])) / len(e_column[self.e_label == 1]))

        plt.plot((b_e[:-1] + 0.5 * (b_e[1:] - b_e[:-1]))[n_e > 0], n_e[n_e > 0], 'o', color='blue', alpha=1,
                 label='Electron Data',
                 markersize=7)

        plt.hist(mc_column[self.mc_label == 2], bins=bins, histtype='stepfilled',  linestyle='dashed', label='Pion MC', color='red',
                 range=range,
                 weights=np.ones(len(mc_column[self.mc_label == 2]))/len(mc_column[self.mc_label == 2]),
                 linewidth=3,
                 alpha=0.5,
                 log=log)

        plt.hist(pi_column[self.pi_label == 2]*pi_scale, bins=bins,histtype='step', alpha=1,
                 # label='Pion Data',
                 color='red', range=range, linewidth=2,
                 weights=np.ones(len(pi_column[self.pi_label == 2]))/len(pi_column[self.pi_label == 2]),
                 log=log)

        n_pi, b_pi = np.histogram(pi_column[self.pi_label == 2] * pi_scale, bins=bins, range=[ll, ul],
                                  weights=np.ones(len(pi_column[self.pi_label == 2])) / len(
                                      pi_column[self.pi_label == 2]))

        plt.plot((b_pi[:-1] + 0.5 * (b_pi[1:] - b_pi[:-1]))[n_pi > 0], n_pi[n_pi > 0], 'o', color='red', alpha=1,
                 label='Pion Data',
                 markersize=7)


        plt.tick_params(labelsize=label_size+2)
        plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.95), fontsize=label_size-1)
        plt.xlabel(column.replace('_', ' '),fontsize=label_size+2)
        plt.ylabel('# [Normalized]', fontsize=label_size+2)

        if y_ul!=None:
            plt.ylim(0,y_ul)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, column + 'beam_{}.png'.format(self.source)))
        # plt.show()
        plt.close(fig)


class BDT_VAR_PLOT():
    def __init__(self, file_path, save_dir, source):

        self.file_path=file_path
        self.save_dir = save_dir

        self.source=source

        self.beam_var = None


    def load(self):
        self.beam_var = pd.read_csv(self.file_path)

        self.label = self.beam_var['Particle_label']


    def compare(self, column, bins, ll=None, ul=None, log=False, y_ul=None):


        source_dict={
            'tb': 'Data samples',
            'mc': 'MC samples',
        }

        var=self.beam_var[column].values

        if ll==None:
            ll=np.amin(var[self.label!=-1])

        if ul==None:
            ul=np.amax(var[self.label!=-1])

        range=[ll, ul]

        fig = plt.figure(figsize=(8, 7))

        ax = plt.gca()

        label_size=18
        plt.text(0.15, 0.9, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.text(0.15, 0.8, source_dict.get(self.source), fontsize=label_size, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, )

        plt.hist(var[self.label == 0], bins=bins, histtype='step', label='Muon', color='green',
                 range=range,
                 weights=np.ones(len(var[self.label == 0])) / len(var[self.label == 0]),
                 linewidth=4,
                 log=log)

        # n_mu, b_mu = np.histogram(var[self.label == 0], bins=bins, range=[ll, ul],
        #                           weights=np.ones(len(var[self.label == 0])) / len(
        #                               var[self.label == 0]))
        #
        # plt.plot((b_mu[:-1] + 0.5 * (b_mu[1:] - b_mu[:-1]))[n_mu > 0], n_mu[n_mu > 0], 'o', color='green', alpha=1,
        #          label='Muon', markersize=7)

        plt.hist(var[self.label == 1], bins=bins, histtype='step', label='Electron', color='blue',
                 range=range,
                 weights=np.ones(len(var[self.label == 1])) / len(var[self.label == 1]),
                 linewidth=4,
                 log=log)

        # n_e, b_e = np.histogram(var[self.label == 1], bins=bins, range=[ll, ul],
        #                           weights=np.ones(len(var[self.label == 1])) / len(
        #                               var[self.label == 1]))
        #
        # plt.plot((b_e[:-1] + 0.5 * (b_e[1:] - b_e[:-1]))[n_e > 0], n_e[n_e > 0], 'o', color='blue', alpha=1,
        #          label='Electron Data',
        #          markersize=7)

        plt.hist(var[self.label == 2], bins=bins, histtype='step', label='Pion', color='red',
                 range=range,
                 weights=np.ones(len(var[self.label == 2])) / len(var[self.label == 2]),
                 linewidth=4,
                 log=log)

        # n_pi, b_pi = np.histogram(var[self.label == 1], bins=bins, range=[ll, ul],
        #                         weights=np.ones(len(var[self.label == 1])) / len(
        #                             var[self.label == 1]))
        # plt.plot((b_pi[:-1] + 0.5 * (b_pi[1:] - b_pi[:-1]))[n_pi > 0], n_pi[n_pi > 0], 'o', color='red', alpha=1,
        #          label='Pion Data',
        #          markersize=7)

        plt.tick_params(labelsize=label_size+2)
        plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.95), fontsize=label_size)
        plt.xlabel(column.replace('_', ' '), fontsize=label_size+2)
        plt.ylabel('# [Normalized]', fontsize=label_size+2)
        if y_ul!=None:
            plt.ylim(0, y_ul)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, column + '_{}.png'.format(self.source)))
        # plt.show()
        plt.close(fig)





def main_compare():
    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/120GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_120_run268_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_120_run236_2023/bdt_var.csv',
        ep=120,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/120GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/100GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_100_run267_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_100_run230_2023/bdt_var.csv',
        ep=100,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/100GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/80GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_80_run266_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_80_run220_2023/bdt_var.csv',
        ep=80,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/80GeV_compare'
        )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/60GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_60_run265_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_60_run216_2023/bdt_var.csv',
        ep=60,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/60GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/50GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv',
        ep=50,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/50GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/30GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_30_run274_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_30_run250_2023/bdt_var.csv',
        ep=30,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/30GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=100)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100,)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/10GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_10_run276_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/10GeV_pi/bdt_var.csv',
        ep=10,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/10GeV_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=90)
    cpr.compare(column='Shower_layer_ratio', bins=50)
    cpr.compare(column='E_dep', bins=100)
    cpr.compare(column='Shower_start', bins=42)
    cpr.compare(column='Shower_length', bins=42)
    cpr.compare(column='Shower_radius', bins=100)
    cpr.compare(column='Hits_no', bins=100)


def compare_e_dep_scale():

    ratio = pd.read_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio.csv')
    e_ratio = ratio['mc_e_peak'] / ratio['e_peak']
    p_ratio = ratio['mc_pi_peak'] / ratio['pi_peak']


    ep_list=list(ratio['ep'].values)

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/120GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_120_run268_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_120_run236_2023/bdt_var.csv',
        ep=120,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/120GeV_compare'
    )

    cpr.load()

    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(120)], pi_scale=p_ratio[ep_list.index(120)])


    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/100GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_100_run267_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_100_run230_2023/bdt_var.csv',
        ep=100,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/100GeV_compare'
    )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(100)], pi_scale=p_ratio[ep_list.index(100)])

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/80GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_80_run266_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_80_run220_2023/bdt_var.csv',
        ep=80,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/80GeV_compare'
        )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(80)], pi_scale=p_ratio[ep_list.index(80)])

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/60GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_60_run265_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_60_run216_2023/bdt_var.csv',
        ep=60,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/60GeV_compare'
    )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(60)], pi_scale=p_ratio[ep_list.index(60)])

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/50GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv',
        ep=50,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/50GeV_compare'
    )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(50)], pi_scale=p_ratio[ep_list.index(50)])

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/30GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_30_run274_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_30_run250_2023/bdt_var.csv',
        ep=30,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/30GeV_compare'
    )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(30)], pi_scale=p_ratio[ep_list.index(30)])

    cpr = Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/10GeV/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_10_run276_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/10GeV_pi/bdt_var.csv',
        ep=10,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/10GeV_compare'
    )

    cpr.load()
    cpr.compare(column='E_dep', bins=100, e_scale=e_ratio[ep_list.index(10)], pi_scale=p_ratio[ep_list.index(10)])

def main_full_compare():

    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/120GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_120_run268_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_120_run236_2023/bdt_var.csv',
    #     ep=120,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/120GeV_full_compare'
    # )
    #
    #
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, y_ul=0.2)
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200)
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20,  y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=30, ul=900, log=False, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=30, ul=4.5,y_ul=1.2 )
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30)


    cpr = Full_Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/100GeV/bdt_var.csv',
        mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_100_run267_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_100_run230_2023/bdt_var.csv',
        ep=100,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/100GeV_full_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=50, )
    cpr.load()
    cpr.compare(column='Shower_layer_ratio', bins=25, )
    cpr.load()
    cpr.compare(column='E_dep', bins=50, ul=1200, )
    cpr.load()
    cpr.compare(column='Shower_start', bins=20, )
    cpr.load()
    cpr.compare(column='Shower_length', bins=20, )
    cpr.load()
    cpr.compare(column='Hits_no', bins=50, ul=450, log=False)
    cpr.load()
    cpr.compare(column='Shower_radius', bins=50, ul=4.5, )
    cpr.load()
    cpr.compare(column='Z_width', bins=30)
    cpr.load()
    cpr.compare(column='FD_1', bins=30, ul=0.6)
    cpr.load()
    cpr.compare(column='FD_6', bins=30, ul=1)
    cpr.load()
    cpr.compare(column='layers_fired', bins=30)
    cpr.load()
    cpr.compare(column='Shower_end', bins=30)


    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/80GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_80_run266_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_80_run220_2023/bdt_var.csv',
    #     ep=80,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/80GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, y_ul=0.4, ll=2)
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, y_ul=0.6)
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1800, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20,y_ul=1 )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20,y_ul=0.5 )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=450, log=False, y_ul=0.8)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, y_ul=0.6)
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30, y_ul=0.6)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30, ll=0.6, y_ul=0.5)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30, y_ul=0.4)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30, y_ul=0.6)
    #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/60GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_60_run265_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_60_run216_2023/bdt_var.csv',
    #     ep=60,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/60GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, )
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, )
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20, )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20, )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=450, log=False)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, )
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30)
    #
    #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/50GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv',
    #     ep=None,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/50GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, y_ul=0.6)
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20,y_ul=1.2 )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=30, ul=450, log=False,y_ul=1.2, x_label='Hits number')
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=30, ul=4.5, y_ul=1.2)
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30, ll=7, ul=35, y_ul=0.7, l_x=0.95)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30, y_ul=0.8, ul=1, x_label=r'$FD_1$')
    # cpr.load()
    # cpr.compare(column='FD_2', bins=30, x_label=r'$FD_2$')
    # cpr.load()
    # cpr.compare(column='FD_3', bins=30,  x_label=r'$FD_3$')
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30, ll=1.18, ul=1.6, y_ul=0.9, x_label=r'$FD_6$')
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=20, ll=18, y_ul=1, x_label='Fired layers')
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=15, ll=15, ul=42)
    # cpr.load()
    # cpr.compare(column='Shower_layer', bins=20, ll=0, ul=40, x_label='Shower layers')
    #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/30GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_30_run274_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_30_run250_2023/bdt_var.csv',
    #     ep=30,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/30GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, )
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, )
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20, )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20, )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=450, log=False)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, )
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30)
    #
    #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/10GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_10_run276_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/10GeV_pi/bdt_var.csv',
    #     ep=10,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/10GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, )
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, )
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=300, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20, )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20, )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=150, log=False)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, )
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30)
    #
    #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/5GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/5GeV_e_Run133/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/5GeV_pi_Run123/bdt_var.csv',
    #     ep=5,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/5GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, )
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, )
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=20, )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=20, )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=450, log=False)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, )
    # cpr.load()
    # cpr.compare(column='Z_width', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_1', bins=30)
    # cpr.load()
    # cpr.compare(column='FD_6', bins=30)
    # cpr.load()
    # cpr.compare(column='layers_fired', bins=30)
    # cpr.load()
    # cpr.compare(column='Shower_end', bins=30)
    #

def main_plot_beam_mc_compare_ann():

    ep_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
    # ep_list=[10 , 20, 30, 40, 50, 60, 70, 80, 100, 120]

    for source in ['tb', 'mc']:
        for ep in ep_list:
            ann_file_path=os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022',
                                       'pi_v3_{}/{}GeV.csv'.format(source, ep))

            mc_file_path= '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/{}GeV/bdt_var.csv'.format(ep)

            cpr = Beam_DATA_COMPARE(
                mc_var_path=mc_file_path,
                mu_beam_var_path=ann_file_path,
                e_beam_var_path=ann_file_path,
                pi_beam_var_path=ann_file_path,
                ep=None,
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}'.format(ep, source),
                source=source
            )

            cpr.load()
            cpr.compare(column='Shower_density', bins=100, )
            cpr.compare(column='Shower_layer_ratio', bins=50, )
            cpr.compare(column='E_dep', bins=100, ul=1700, )
            cpr.compare(column='Shower_start', bins=42, )
            cpr.compare(column='Shower_length', bins=42, )
            cpr.compare(column='Hits_no', bins=100, ul=600)
            cpr.compare(column='Shower_radius', bins=100, ul=4.5, )


def main_plot_beam_mc_compare_ann_pion_only():

    # ep_list=[10, 30,  50,  60, 80, 100, 120]
    ep_list = [ 50]
    # ep_list=[10 , 20, 30, 40, 50, 60, 70, 80, 100, 120]

    # for source in ['mc']:
    for source in [ 'tb']:
        for ep in ep_list:
            # ann_file_path=os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022',
            #                            'pi_v3_{}_0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1/{}GeV.csv'.format(source, ep))
            ann_file_path = os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022',
                                         'pi_v3_{}/{}GeV.csv'.format(
                                             source, ep))

            mc_file_path= '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/{}GeV/bdt_var.csv'.format(ep)

            cpr = Beam_DATA_COMPARE(
                mc_var_path=mc_file_path,
                mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
                e_beam_var_path=glob.glob('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_{}_*/bdt_var.csv'.format(ep))[0],
                pi_beam_var_path=ann_file_path,
                ep=None,
                # save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}_pi_only_pi_v3_mc_0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1'.format(ep, source),
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}_pi_only'.format(
                    ep, source),
                source=source
            )


            cpr.compare(column='Shower_density', bins=50, ul=5, y_ul=0.7 )
            cpr.compare(column='Shower_layer_ratio', bins=25,y_ul=1.2 )
            cpr.compare(column='E_dep', bins=50, ul=1700, )
            cpr.compare(column='Shower_start', bins=20, y_ul=1.2)
            cpr.compare(column='Shower_length', bins=20, y_ul=1.2)
            cpr.compare(column='Hits_no', bins=30, ul=500, y_ul=1.2)
            cpr.compare(column='Shower_radius', bins=30, ul=4.5, y_ul=1.2)

def main_plot_beam_mc_compare_ann_pion_only_mc_v2():

    # ep_list=[10, 30,  50,  60, 80, 100, 120]
    ep_list = [50]
    # ep_list=[10 , 20, 30, 40, 50, 60, 70, 80, 100, 120]

    for source in ['mc']:
    # for source in ['mc', 'tb']:
        for ep in ep_list:
            ann_file_path=os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022',
                                       'pi_v3_{}_0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1/{}GeV.csv'.format(source, ep))
            # ann_file_path = os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022',
            #                              'pi_v3_{}/{}GeV.csv'.format(
            #                                  source, ep))

            mc_file_path= '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/{}GeV/bdt_var.csv'.format(ep)

            cpr = Beam_DATA_COMPARE(
                mc_var_path=mc_file_path,
                mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
                e_beam_var_path=glob.glob('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_{}_*/bdt_var.csv'.format(ep))[0],
                pi_beam_var_path=ann_file_path,
                ep=None,
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}_pi_only_pi_v3_mc_0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1'.format(ep, source),
                # save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}_pi_only'.format(
                #     ep, source),
                source=source
            )

            cpr.compare(column='Shower_density', bins=50, ul=5, y_ul=0.7)
            cpr.compare(column='Shower_layer_ratio', bins=25, y_ul=1.2)
            cpr.compare(column='E_dep', bins=50, ul=1700, )
            cpr.compare(column='Shower_start', bins=20, y_ul=1.2)
            cpr.compare(column='Shower_length', bins=20, y_ul=1.2)
            cpr.compare(column='Hits_no', bins=30, ul=500, y_ul=1.2)
            cpr.compare(column='Shower_radius', bins=30, ul=4.5, y_ul=1.2)

def main_dataset_compare():
    cpr = Full_Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/0515_no_noise_v2/Train/bdt_var.csv',
        mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        ep=80,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/data_mc_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=20, )
    cpr.load()
    cpr.compare(column='Shower_layer_ratio', bins=25, )
    cpr.load()
    cpr.compare(column='E_dep', bins=20, ul=1200, )
    cpr.load()
    cpr.compare(column='Shower_start', bins=20, )
    cpr.load()
    cpr.compare(column='Shower_length', bins=20, )
    cpr.load()
    cpr.compare(column='Hits_no', bins=20, ul=450, log=False)
    cpr.load()
    cpr.compare(column='Shower_radius', bins=30, ul=4.5, )

    cpr = Full_Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/0515_no_noise_v2/Train/bdt_var.csv',
        mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Train/bdt_var.csv',
        ep=80,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/data_mc_compare'
    )

    cpr.load()
    cpr.compare(column='Shower_density', bins=20, )
    cpr.load()
    cpr.compare(column='Shower_layer_ratio', bins=25, )
    cpr.load()
    cpr.compare(column='E_dep', bins=20, ul=1200, )
    cpr.load()
    cpr.compare(column='Shower_start', bins=20, )
    cpr.load()
    cpr.compare(column='Shower_length', bins=20, )
    cpr.load()
    cpr.compare(column='Hits_no', bins=20, ul=450, log=False)
    cpr.load()
    cpr.compare(column='Shower_radius', bins=30, ul=4.5, )

if __name__ == '__main__':
    # main_compare()
    # main_full_compare()

    cpr = Full_Shower_Compare(
        mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515_e_pi/50GeV/bdt_var.csv',
        mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
        e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
        pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv',
        ep=None,
        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/50GeV_full_compare'
    )
    #
    cpr.load()
    cpr.compare(column='Shower_density', bins=50, y_ul=0.2, ll=2, ul=5)
    cpr.load()
    cpr.compare(column='Shower_layer_ratio', bins=44, y_ul=0.25)
    cpr.load()
    cpr.compare(column='E_dep', bins=50, ul=1200, y_ul=1.2)
    cpr.load()
    cpr.compare(column='Shower_start', bins=20, y_ul=1, ll=0, ul=45)
    cpr.load()
    cpr.compare(column='Shower_length', bins=20, y_ul=0.2, ll=0, ul=40)
    cpr.load()
    cpr.compare(column='Hits_no', bins=30, ll=100, ul=450, log=False, y_ul=0.6, x_label='Hits Number')
    cpr.load()
    cpr.compare(column='Shower_radius', bins=30, ll=1, ul=4.5, y_ul=0.4)
    cpr.load()
    cpr.compare(column='Z_width', bins=30, ll=5, ul=35, y_ul=0.6, l_x=0.95, x_label='Z Depth')
    cpr.load()
    cpr.compare(column='FD_1', bins=30, y_ul=0.25, ul=1, ll=0.6, x_label=r'$\mathrm{FD}_1$')
    cpr.load()
    cpr.compare(column='FD_2', bins=30, x_label=r'$FD_2$')
    cpr.load()
    cpr.compare(column='FD_3', bins=30, x_label=r'$FD_3$')
    cpr.load()
    cpr.compare(column='FD_6', bins=30, ll=1.1, ul=1.6, y_ul=0.3, x_label=r'$\mathrm{FD}_6$')
    cpr.load()
    cpr.compare(column='layers_fired', bins=20, ll=15, ul=45, y_ul=1, x_label='Fired Layers')
    cpr.load()
    cpr.compare(column='Shower_end', bins=15, ll=15, ul=45, y_ul=0.6)
    cpr.load()
    cpr.compare(column='Shower_layer', bins=20, ll=0, ul=45, y_ul=0.6, x_label='Shower Layers')
    # compare_e_dep_scale()
    #
    #
    # main_dataset_compare()
    #
    # main_plot_beam_mc_compare_ann_pion_only()
    # main_plot_beam_mc_compare_ann_pion_only_mc_v2()

    # #
    # cpr = Full_Shower_Compare(
    #     mc_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/50GeV/bdt_var.csv',
    #     mu_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/100GeV_mu_Run25/bdt_var.csv',
    #     e_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
    #     pi_beam_var_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv',
    #     ep=None,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/50GeV_full_compare'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, )
    # cpr.load()
    # cpr.compare(column='Shower_layer_ratio', bins=25, )
    # cpr.load()
    # cpr.compare(column='E_dep', bins=50, ul=1200, )
    # cpr.load()
    # cpr.compare(column='Shower_start', bins=40, )
    # cpr.load()
    # cpr.compare(column='Shower_length', bins=40, )
    # cpr.load()
    # cpr.compare(column='Hits_no', bins=50, ul=450, log=False)
    # cpr.load()
    # cpr.compare(column='Shower_radius', bins=50, ul=4.5, )

    # cpr=BDT_VAR_PLOT(
    #     file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_0720/Train/bdt_var.csv',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/bdt_tb_var_0720',
    #     source='tb'
    #
    # )
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, y_ul=0.6)
    # cpr.compare(column='Shower_layer_ratio', bins=25,y_ul=1.2)
    # cpr.compare(column='E_dep', bins=50, ul=1700, y_ul=1.2)
    # cpr.compare(column='Shower_start', bins=20, y_ul=1.2)
    # cpr.compare(column='Shower_length', bins=20, y_ul=1.2)
    # cpr.compare(column='Hits_no', bins=30, ul=600, log=False,y_ul=0.9)
    # cpr.compare(column='Shower_radius', bins=30, ul=4.5, y_ul=1.1)
    #
    # cpr = BDT_VAR_PLOT(
    #     file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/0720_version/Train/bdt_var.csv',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/bdt_mc_var_0720',
    #     source='mc'
    #
    # )
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=50, y_ul=0.7)
    # cpr.compare(column='Shower_layer_ratio', bins=25, y_ul=1.2)
    # cpr.compare(column='E_dep', bins=50, ul=1700, y_ul=1.2)
    # cpr.compare(column='Shower_start', bins=20, y_ul=1.2)
    # cpr.compare(column='Shower_length', bins=20, y_ul=1.2)
    # cpr.compare(column='Hits_no', bins=30, ul=600, log=False, y_ul=0.8)
    # cpr.compare(column='Shower_radius', bins=30, ul=4.5, y_ul=1.1)

    # main_plot_beam_mc_compare_ann()


    # ann_file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/temp/tb_test/bdt_var.csv'
    #
    # mc_file_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_ckv_0615/Test/bdt_var.csv'
    #
    # cpr = Beam_DATA_COMPARE(
    #     mc_var_path=mc_file_path,
    #     mu_beam_var_path=ann_file_path,
    #     e_beam_var_path=ann_file_path,
    #     pi_beam_var_path=ann_file_path,
    #     ep=None,
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/temp/tb_test',
    #     source='tb'
    # )
    #
    # cpr.load()
    # cpr.compare(column='Shower_density', bins=100, )
    # cpr.compare(column='Shower_layer_ratio', bins=50, )
    # cpr.compare(column='E_dep', bins=100, ul=1700, )
    # cpr.compare(column='Shower_start', bins=42, )
    # cpr.compare(column='Shower_length', bins=42, )
    # cpr.compare(column='Hits_no', bins=100, ul=600)
    # cpr.compare(column='Shower_radius', bins=100, ul=4.5, )
    pass
