import numpy as np
import uproot
from ReadRoot import *
import os

def makeDatasets(file_path, save_path):
    '''
    1: inout: root file
    2: output: numpy array NCHW (,40,18,18)
    '''
    # read raw root file
    hcal_energy, x, y, z = readRootFileCell(file_path)
    num_events = len(hcal_energy)
    assert num_events == len(x)
    assert num_events == len(y)
    assert num_events == len(z)
    # NHWC
    depoits = np.zeros((num_events, 18, 18, 40))
    for i in range(num_events):
        energies_ = hcal_energy[i]
        x_ = ((x[i] + 340) / 40).astype(int)
        y_ = ((y[i] + 340) / 40).astype(int)
        z_ = ((z[i] - 301.5) / 25).astype(int)
        num_events_ = len(energies_)
        assert num_events_ == len(x_)
        assert num_events_ == len(y_)
        assert num_events_ == len(z_)
        for j in range(num_events_):
            depoits[i, x_[j], y_[j], z_[j]] += energies_[j]
    # NCHW
    # depoits = np.transpose(depoits, (0, 3, 1, 2))
    np.save(save_path, depoits)

def makeLabels(mu_nums,e_nums,pi_nums,save_path):
    '''

    :param mu_nums:
    :param e_nums:
    :param pi_nums:
    :param save_path:
    :return:
    mu+:0
    e++:1
    pi+:2
    '''
    mu_labels=np.zeros(mu_nums)
    e_labels=np.ones(e_nums)
    pi_labels=np.ones(pi_nums)*2
    labels=np.append(mu_labels,e_labels)
    labels=np.append(labels,pi_labels).astype(np.longlong)
    np.save(save_path,labels)

def combineDatasets(name_lists,shuffle,load_path,save_path):
    '''
    load_path:xxx/xx{}.npy
    '''
    datasets=[]

    for index,name in enumerate(name_lists):
        file_name=load_path.format(name)
        data=np.load(file_name)
        if index==0:
            datasets=data
        datasets=np.append(datasets,data,axis=0)
    if shuffle:
        np.random.shuffle(datasets)
    np.save(save_path,datasets)


def splitDatasets(file_path,save_dir,indices_or_sections,shuffle):
    '''
    :param file_path:
    :param save_dir:
    :param indices_or_sections: a list, value in it means the ratio of the split, e+.g.[0.1,0.2]=1:1:8
    :param shuffle:
    :return:
    '''
    data=np.load(file_path)
    if shuffle:
        np.random.shuffle(data)
    num=len(data)
    indices_or_sections=list(map(lambda x: int(x*num),indices_or_sections))
    results=np.split(data,indices_or_sections=indices_or_sections,axis=0)
    for i ,result in enumerate(results):
        save_path=os.path.join(save_dir,'{}.npy'.format(i))
        np.save(save_path,result)

def makeFinalDatasets(file_path_list,save_dir):
    '''

    :param file_path_list: order should be mu, e+, pion
    :param shuffle:
    :param save_dir:
    :return:
    '''
    datasets = []
    data_length=[]
    for index, file_path in enumerate(file_path_list):
        data = np.load(file_path)
        data_length.append(len(data))
        if index == 0:
            datasets = data
        else:
            datasets = np.append(datasets, data, axis=0)
    datasets_save_path=os.path.join(save_dir,'datasets.npy')
    labels_save_path=os.path.join(save_dir,'labels.npy')
    np.save(datasets_save_path, datasets)
    makeLabels(mu_nums=data_length[0],e_nums=data_length[1],pi_nums=data_length[2],save_path=labels_save_path)

if __name__ == '__main__':
    file_path_list=['/Users/songsiyuan/PID/Model/Data/File/test/mu+/split/2.npy'
        ,'/Users/songsiyuan/PID/Model/Data/File/test/e+/split/2.npy'
        ,'/Users/songsiyuan/PID/Model/Data/File/test/pi+/split/2.npy',]
    save_path = '/Users/songsiyuan/PID/Model/Data/File/test/test'

    makeFinalDatasets(file_path_list,save_path)

    path='/Users/songsiyuan/PID/Model/Data/File/test/test/labels.npy'
