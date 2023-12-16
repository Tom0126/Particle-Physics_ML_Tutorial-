import numpy as np
import matplotlib.pyplot as plt

def plotROC(fpr_path,tpr_path,auroc_path,signal,save_path,data_type):

    particle_dim={'mu+':0,'e+':1,'pi+':2,'noise':3}
    particle_name={'mu+':r'$\mu^+$','e+':r'$e^+$','pi+':r'$\pi^+$','noise':'Noise'}

    text_dict = {
        'mc': 'Simulation',
        'data': 'Test beam data'
    }

    fprs=np.load(fpr_path, allow_pickle=True)
    tprs=np.load(tpr_path, allow_pickle=True)
    auroc=np.load(auroc_path, allow_pickle=True)

    fpr=fprs[particle_dim.get(signal)]
    tpr = tprs[particle_dim.get(signal)]
    auc=auroc[particle_dim.get(signal)]



    fig=plt.figure(figsize=(6, 5))
    ax=plt.gca()
    plt.plot(tpr,1-fpr,label='ANN',color='red')

    plt.xlabel('Signal efficiency',fontsize=10)
    plt.ylabel('Background rejection rate',fontsize=10)

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.text(0.1, 0.84, text_dict.get(data_type,''), fontsize=12, fontstyle='normal',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.text(0.1, 0.78, '{} Signals'.format(particle_name.get(signal)), fontsize=12, fontstyle='normal',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.text(0.1, 0.72, 'ANN AUC = {:.3f}'.format(auc), fontsize=12, fontstyle='normal',horizontalalignment='left',
             verticalalignment='center',transform=ax.transAxes,)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5, alpha=0.5)

    plt.ylim(0,1.45)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0, 1, 11))

    plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes,fontsize=13)
    plt.savefig(save_path.format(signal))
    plt.close(fig)
    # plt.show()
def plot_s_b_threshold(fpr_path,tpr_path,signal,save_path,threshold_num, data_type):
    particle_dim={'mu+':0,'e+':1,'pi+':2,'noise':3}
    particle_name = {'mu+': r'Muon', 'e+': r'Electron', 'pi+': r'Pion', 'noise': 'Noise'}

    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'data': 'Data test set\nData training approach'
    }

    fprs=np.load(fpr_path, allow_pickle=True)
    tprs=np.load(tpr_path, allow_pickle=True)


    fpr=fprs[particle_dim.get(signal)]
    tpr = tprs[particle_dim.get(signal)]
    bkr=1-fpr

    thresholds=np.linspace(1,0,threshold_num)

    assert len(tpr) == threshold_num

    fig=plt.figure(figsize=(8, 7))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111)
    l1=ax.plot(thresholds[::5],tpr[::5],'o',label=particle_name.get(signal),color='red', markersize=6)
    ax2=ax.twinx()
    l2=ax2.plot(thresholds[::5], bkr[::5],'^',label='Backgrounds', color='black', markersize=6)

    ax.set_xlabel('ANN probability threshold',fontsize=14)
    ax.set_ylabel('{} efficiency'.format(particle_name.get(signal))+r'$(N_{S}^{sel.}/N_{S})$',fontsize=14)
    ax2.set_ylabel('Bkg. rejection rate '+r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=14)

    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',horizontalalignment='left',
             verticalalignment='top',transform=ax.transAxes,)
    plt.text(0.1, 0.89, text_dict.get(data_type,''), fontsize=14, fontstyle='normal',horizontalalignment='left',
             verticalalignment='top',transform=ax.transAxes,)

    # plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.tick_params(labelsize=14, direction='in', length=5)
    ax2.tick_params(labelsize=14, direction='in', length=5)

    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0.9, 1, 6))
    ax2.set_yticks(np.linspace(0.9, 1, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.02)
    ax2.set_ylim(0.9, 1.02)

    plt.minorticks_on()

    ax.tick_params(which='minor', direction='in', length=3)
    ax2.tick_params(which='minor', direction='in', length=3)

    ax.set_xticks(np.linspace(0, 1, 51), minor=True)
    ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)
    ax2.set_yticks(np.linspace(0.9, 1, 26), minor=True)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.9,0.98),fontsize=14)


    # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
    plt.savefig(save_path.format(signal))
    plt.close(fig)


def plot_s_b_ratio_threshold(fpr_path,tpr_path,signal,save_path,threshold_num, data_type):

    label_size=18
    particle_dim={'mu+':0,'e+':1,'pi+':2,'noise':3}
    particle_name = {'mu+': r'Muon', 'e+': r'Electron', 'pi+': r'Pion', 'noise': 'Noise'}
    y_lim = {'mu+': 4000, 'e+': 6000, 'pi+': 5000, 'noise': 3000}
    text_dict = {
        'mc': 'MC test set\nMC training approach',
        'data': 'Data test set\nData training approach'
    }

    fprs=np.load(fpr_path, allow_pickle=True)
    tprs=np.load(tpr_path, allow_pickle=True)


    fpr=fprs[particle_dim.get(signal)]
    tpr = tprs[particle_dim.get(signal)]
    bkr=1/fpr

    thresholds=np.linspace(1,0,threshold_num)

    assert len(tpr) == threshold_num

    fig=plt.figure(figsize=(8, 7))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111)
    l1=ax.plot(thresholds[::5],tpr[::5],'o',label=particle_name.get(signal),color='red', markersize=6)
    ax2=ax.twinx()
    l2=ax2.plot(thresholds[::5], bkr[::5],'^',label='Backgrounds', color='black', markersize=6)

    ax.set_xlabel('ANN {} likelihood threshold'.format((particle_name.get(signal)).lower()),fontsize=label_size)
    ax.set_ylabel('{} efficiency'.format(particle_name.get(signal))+r'$(N_{S}^{sel.}/N_{S})$',fontsize=label_size-2)
    ax2.set_ylabel('Bkg. rejection '+r'$N_{B}/(N_{B}^{sel.}$)', fontsize=label_size-2)

    plt.text(0.1, 0.95, 'CEPC AHCAL', fontsize=label_size, fontstyle='oblique', fontweight='bold',horizontalalignment='left',
             verticalalignment='top',transform=ax.transAxes,)
    plt.text(0.1, 0.89, text_dict.get(data_type,''), fontsize=label_size, fontstyle='normal',horizontalalignment='left',
             verticalalignment='top',transform=ax.transAxes,)

    # plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.tick_params(labelsize=label_size-2, direction='in', length=5)
    ax2.tick_params(labelsize=label_size-2, direction='in', length=5)

    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0.9, 1, 6))
    # ax2.set_yticks(np.linspace(0.9, 1, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.03)
    # ax2.set_ylim(1, 4*np.amax(bkr[::5][~np.isinf(bkr[::5])]))
    ax2.set_ylim(1, y_lim.get(signal))

    plt.minorticks_on()

    ax.tick_params(which='minor', direction='in', length=3)
    ax2.tick_params(which='minor', direction='in', length=3)

    ax.set_xticks(np.linspace(0, 1, 51), minor=True)
    ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)
    # ax2.set_yticks(np.linspace(0.9, 1, 26), minor=True)
    ax2.set_yscale('log')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.95,0.98),fontsize=label_size-2)


    # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
    plt.savefig(save_path.format((particle_name.get(signal)).lower()))
    plt.close(fig)

def plot_s_b_ep(threshold, tpr_file_lists, fpr_file_lists,ep_lists,signal, save_path, threshold_num):
    particle_dim = {'mu+': 0, 'e+': 1, 'pi+': 2, 'noise': 3}
    particle_name = {'mu+': r'$\mu^+$', 'e+': r'$e^+$', 'pi+': r'$\pi^+$', 'noise': 'Noise'}

    tpr=[]
    bkr=[]

    assert len(ep_lists) == len(tpr_file_lists)
    assert len(ep_lists) == len(fpr_file_lists)

    for tpr_path, fpr_path in zip(tpr_file_lists,fpr_file_lists):
        tpr_=np.load(tpr_path)
        fpr_=np.load(fpr_path)

        tpr.append(tpr_[particle_dim.get(signal),-1*int(threshold*(threshold_num))])
        bkr.append(1-fpr_[particle_dim.get(signal), -1*int(threshold * (threshold_num))])
        tpr_ = None
        fkr_ = None


    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    l1 = ax.plot(ep_lists, tpr, 'o', label=particle_name.get(signal), color='red')
    ax2 = ax.twinx()
    l2 = ax2.plot(ep_lists, bkr, '^', label='Backgrounds', color='black')

    ax.set_xlabel('Energy [GeV]', fontsize=10)
    ax.set_ylabel('{} efficiency'.format(particle_name.get(signal)), fontsize=10)
    ax2.set_ylabel('Background rejection rate', fontsize=10)

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.84, 'AHCAL PID Threshold = {}'.format(threshold), fontsize=12, fontstyle='normal', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    ax.set_xticks(ep_lists)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax2.set_yticks(np.linspace(0, 1, 11))

    ax.set_ylim(0, 1.3)
    ax2.set_ylim(0, 1.3)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right')

    # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
    plt.savefig(save_path.format(signal))
    plt.close(fig)

if __name__=='__main__':
    fpr_path='../roc/fpr.npy'
    tpr_path = '../roc/tpr.npy'
    auroc_path = '../roc/auroc.npy'
    bdt_path='../roc/pion_roc_bdt.txt'
    save_path='Fig/ann_bdt_compare.png'
    plotROC(fpr_path=fpr_path,tpr_path=tpr_path,auroc_path=auroc_path,signal='pi+',save_path=save_path)