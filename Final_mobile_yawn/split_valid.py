from shutil import copyfile
import os
from sklearn.model_selection import train_test_split

def copy_pictures():
    for i in range(2):
        src = 'train_temp/%d/'%i
        for root,subdir,files in os.walk(src):
            X_train, X_valid = train_test_split(files, test_size=0.18, 
                                               random_state=1234,
                                               shuffle=True)
            dataset = [X_train, X_valid]
            idx = 0
            for sets in ['train', 'valid']:
                dst = sets+'/%d/'%i
                if not os.path.exists(dst):
                    os.mkdir(dst)
                for file in dataset[idx]:
                    copyfile(src+file, dst+file)
                idx += 1

copy_pictures()
