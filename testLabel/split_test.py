from shutil import copyfile
import os
from sklearn.model_selection import train_test_split

def copy_pictures():
    for i in range(2):
        src = 'deg%d/'%i
        for root,subdir,files in os.walk(src):
            X_train, X_test = train_test_split(files, test_size=0.15, 
                                               random_state=1234,
                                               shuffle=True)
            dataset = [X_train, X_test]
            idx = 0
            #print(dataset[idx])
            #print(X_train)
            for sets in ['train_temp', 'test']:
                dst = sets+'/%d/'%i
                if not os.path.exists(dst):
                    os.mkdir(dst)
                for file in dataset[idx]:
                    copyfile(src+file, dst+file)
                idx += 1

copy_pictures()
