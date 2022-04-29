import os
import pickle
import numpy as np

def folder_name(file_dir):
    for i, (root, dirs, files) in enumerate(os.walk(file_dir)):
        if i == 0 :
            dir_list = []
            for dir in dirs:
                dir_list.append(dir)
        for file in files:
            print(os.path.join(root,file))
            w1, w2, w3, w4, w5, w6 = pickle.load(open(os.path.join(root,file), 'rb'), encoding='utf-8')

            print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
            print('F1 on T: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
            print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
            print('Clusters: %.4f +/- %.4f' % (np.mean(41), np.std(w4)))
            print('')
file_dir = 'outputs'
folder_name(file_dir)