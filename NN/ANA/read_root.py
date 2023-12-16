#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 20:02
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : read_root.py
# @Software: PyCharm

import uproot

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]
if __name__ == '__main__':
    pass
