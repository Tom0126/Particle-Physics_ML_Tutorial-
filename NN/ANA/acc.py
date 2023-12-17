import matplotlib.pyplot as plt
import numpy as np


# combination: 98.86%
# pi_path='../ahcal_beamtest/pi+.npy'
# e_path='../ahcal_beamtest/e+.npy'
# mu_path='../ahcal_beamtest/mu+.npy'

def plotACC(combi_path, mu_path, e_path, pi_path, save_path, n_classes=3, proton_path=None):
    combi_acc = np.load(combi_path)
    e_acc = np.load(e_path)
    pi_acc = np.load(pi_path)
    mu_acc = np.load(mu_path)
    lower_limit = np.min([np.min(pi_acc[1]), np.min(e_acc[1]), np.min(mu_acc[1])])
    upper_limit = np.max([np.max(pi_acc[1]), np.max(e_acc[1]), np.max(mu_acc[1])])
    energy_points = sorted([100, 20, 40, 60, 80, 120, 30, 50, 70, 90, 160])
    plt.figure(figsize=(6, 5))

    if n_classes == 4:  # with proton classfication
        proton_acc = np.load(proton_path)
        lower_limit = np.min([lower_limit, np.min(proton_acc[1])])
        upper_limit = np.max([upper_limit, np.max(proton_acc[1])])
        plt.plot(proton_acc[0], proton_acc[1], 'o', color='darkorange', label='proton', markersize=4)
        
    plt.text(20, 96.6, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold')
    # plt.text(20, 96.3, 'AHCAL PID', fontsize=12, fontstyle='normal')
    # combination acc
    plt.plot(np.linspace(20, 160, 10), combi_acc[0] * np.ones(10), linestyle=':', color='black')
    plt.text(100, combi_acc[0], 'Overall Accuracy: {}%'.format(combi_acc[0]))
    # base acc
    plt.plot(np.linspace(20, 160, 10), lower_limit * np.ones(10), linestyle=':', color='blueviolet')
    plt.text(107, lower_limit, 'Lowest Accuracy: {}%'.format(lower_limit), color='blueviolet')
    plt.plot(np.linspace(20, 160, 10), upper_limit * np.ones(10), linestyle=':', color='blueviolet')
    plt.text(108, upper_limit, 'Highest Accuracy: {}%'.format(upper_limit), color='blueviolet')
    #  e+ acc
    plt.plot(e_acc[0], e_acc[1], 'o', color='blue', label='e+', markersize=4)
    #  p+ acc
    plt.plot(pi_acc[0], pi_acc[1], 'o', color='red', label='pion+', markersize=4)
    #  mu+ acc
    plt.plot(mu_acc[0], mu_acc[1], 'o', color='green', label='mu+', markersize=4)

    plt.ylim([np.min([lower_limit, 95]), 100.5])
    plt.xticks(energy_points)
    plt.legend(loc='lower right')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy [%]')
    plt.savefig(save_path)
    plt.close()
    # plt.show()

def plotACCbar(acc,save_path, threshold):

    fig=plt.figure(figsize=(6,5))
    ax=plt.gca()
    xticks_dict={
        3:['mu+', 'e+', 'pi+'],
        4:['mu+', 'e+', 'pi+','noise'],
    }
    bar=plt.bar(np.arange(len(acc)),acc,align='center',tick_label=xticks_dict.get(len(acc)),width=0.5)
    plt.bar_label(bar,label_type='edge')
    plt.ylabel('Efficiency [%]')

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.text(0.1, 0.85, 'AHCAL PID Threshold = {}'.format(threshold), fontsize=12, fontstyle='normal',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)

    plt.yticks(np.linspace(0, 100, 11))
    plt.ylim([0, 130])
    plt.savefig(save_path)


def plot_purity_threshold(purities,signal,save_path,data_type,threshold_num):
    particle_dim={'e+':0,'pi+':1}
    particle_name = {'mu+': r'$\mu^+$', 'e+': r'$e^+$', 'pi+': r'$\pi^+$', 'noise': 'Noise'}

    text_dict={
        'mc': 'Simulation',
        'data': 'Test Beam Data'
    }

    thresholds=np.linspace(0,1,threshold_num)
    purity=purities[particle_dim.get(signal)]

    thresholds=thresholds[:-1]
    purity=purity[:-1]

    fig=plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    l1=ax.plot(thresholds[::5],purity[::5],'o',label=particle_name.get(signal),color='red')

    ax.set_xlabel('Threshold',fontsize=10)
    ax.set_ylabel('{} purity'.format(particle_name.get(signal)),fontsize=10)
    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.text(0.1, 0.84, text_dict.get(data_type,''), fontsize=12, fontstyle='normal',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)

    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0, 1, 11))


    ax.set_ylim(0,1.3)
    ax.legend(loc='upper right')

    # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
    plt.savefig(save_path.format(signal))
    plt.close(fig)


def plot_purity_ep(threshold, file_lists, ep_lists,signal, save_path, threshold_num):
    particle_dim = {'mu+': 0, 'e+': 1, 'pi+': 2, 'noise': 3}
    particle_name = {'mu+': r'$\mu^+$', 'e+': r'$e^+$', 'pi+': r'$\pi^+$', 'noise': 'Noise'}



    y=[]

    assert len(ep_lists) == len(file_lists)

    for file_path in file_lists:
        _=np.load(file_path)
        y.append(_[particle_dim.get(signal),int(threshold*(threshold_num-1))])
        _=None



    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(ep_lists, y, 'o', label=particle_name.get(signal), color='red')

    ax.set_xlabel('Energy [GeV]', fontsize=10)
    ax.set_ylabel('{} purity'.format(particle_name.get(signal)), fontsize=10)
    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.84, 'AHCAL PID Threshold = {}'.format(threshold), fontsize=12, fontstyle='normal', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    ax.set_xticks(ep_lists)
    ax.set_yticks(np.linspace(0, 1, 11))

    ax.set_ylim(0, 1.3)
    ax.legend(loc='upper right')

    plt.savefig(save_path.format(signal))
    plt.close(fig)