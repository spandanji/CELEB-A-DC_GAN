# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:40:04 2018

@author: WORKSTATION 3
"""

import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = 'C:\\Spandan\\celebA\\'
save_root = 'C:\\Spandan\\cele\\celebA64x64'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)
save_root = save_root +'\\'
#%%
# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('{} images complete'.format(i))
