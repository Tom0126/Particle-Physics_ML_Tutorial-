import uproot
import numpy as np

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]


def readRootFileCell(path):
    file = uproot.open(path)

    # get keys
    # print(file.keys())

    # A TTree File
    simulation = file['T']

    # Get data->numpy
    data = simulation.arrays(library="np")

    # e+,x,y,z
    hcal_energy = data['hcal_energy']
    x = data['hcal_x']
    y = data['hcal_y']
    z = data['hcal_z']

    return hcal_energy, x, y, z


if __name__ == '__main__':
    file_path = '/Users/songsiyuan/PID/File/e+/hcal_e+_80GeV_2cm_10k.root'
    hcal_energy, x, y, z=readRootFileCell(file_path)
    # print('x_min: {}, x_max: {}'.format(np.min((x[99])/4),np.max((x[99]+34)/4)))
    # print('y_min: {}, y_max: {}'.format(np.min((y[99] + 34) / 4), np.max((y[99] + 34) / 4)))
    # print('z_min: {}, z_max: {}'.format(np.min((z[99] - 301.5) / 25), np.max((z[99]-301.5) / 25)))
    print('x_min: {}, x_max: {}'.format(np.min((x[99] +340)/ 40), np.max((x[99] + 340) / 40)))
    print('y_min: {}, y_max: {}'.format(np.min((y[99] + 340) / 40), np.max((y[99] + 340) / 40)))
    print('z_min: {}, z_max: {}'.format(np.min((z[99] - 301.5) / 25), np.max((z[99]-301.5) / 25)))
    print((z[99] - 301.5) / 25)