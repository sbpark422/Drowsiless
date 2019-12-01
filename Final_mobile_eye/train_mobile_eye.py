# %%
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
plt.style.use('dark_background')
from keras.callbacks import TensorBoard

tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# %%
'''
# Load Dataset
'''

# %%
import glob

# %%
imgs = glob.glob('./mrlEyes_2018_01/*/*.png')

# %%
from sklearn.model_selection import train_test_split

# %%
train_imgs, valid_imgs = train_test_split(imgs, test_size=0.2)

# %%
len(train_imgs), len(valid_imgs)

# %%
from collections import Counter
from PIL import Image

# %%
x_train = np.empty((len(train_imgs), 32, 32, 1))
y_train = np.empty(len(train_imgs))

x_valid = np.empty((len(valid_imgs), 32, 32, 1))
y_valid = np.empty(len(valid_imgs))

# %%
for idx, train in enumerate(train_imgs):
    x_train[idx] = np.expand_dims(np.array(Image.open(train).resize((32, 32), Image.BICUBIC)), -1)
    y_train[idx] = int(train.split('/')[-1].split('_')[4])

# %%
for idx, valid in enumerate(valid_imgs):
    x_valid[idx] = np.expand_dims(np.array(Image.open(valid).resize((32, 32), Image.BICUBIC)), -1)
    y_valid[idx] = int(valid.split('/')[-1].split('_')[4])

# %%
x_train.shape, y_train.shape, x_valid.shape, y_valid.shape

# %%
'''
# Preview
'''

# %%
plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((32, 32)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_valid[4]))
plt.imshow(x_valid[4].reshape((32, 32)), cmap='gray')

# %%
'''
# Data Augmentation
'''

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x=x_train, y=y_train,
    batch_size=32,
    shuffle=True
)

valid_generator = valid_datagen.flow(
    x=x_valid, y=y_valid,
    batch_size=32,
    shuffle=False
)

# %%
'''
# Build Model
'''

# %%
inputs = Input(shape=(32, 32, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

# %%
'''
5개의 layer를 가진 모델
'''

# %%
'''
# Train
'''

# %%
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit_generator(
    train_generator, epochs=50, validation_data=valid_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='valid_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='valid_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05), tb_hist
    ]
)

model.save('models/train_mrl_0.h5')


# %%
#start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#
#model.fit_generator(
#    train_generator, epochs=50, validation_data=valid_generator,
#    callbacks=[
#        ModelCheckpoint('models/%s.h5' % (start_time), monitor='valid_acc', save_best_only=True, mode='max', verbose=1),
#        ReduceLROnPlateau(monitor='valid_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
#    ]
#)

# %%
#model.save('models/train_mrl_0.h5')

# %%
#model.save(‘train_mrl_0.hdf5’)

# %%
#history['loss'], history['val_loss']

# %%
'''
# Confusion Matrix
'''

# %%
#from sklearn.metrics import accuracy_score, confusion_matrix
#import seaborn as sns

#model = load_model('models/%s.h5' % (start_time))

#y_pred = model.predict(x_val/255.)
#y_pred_logical = (y_pred > 0.5).astype(np.int)

#print ('test acc: %s' % accuracy_score(y_val, y_pred_logical))
#cm = confusion_matrix(y_val, y_pred_logical)
#sns.heatmap(cm, annot=True)

# %%
'''
# Distribution of Prediction
'''

# %%
#ax = sns.distplot(y_pred, kde=False)

# %%
'''
---
'''

# %%
'''
# MobileNet V2
'''

# %%
#from keras.applications import MobileNetV2

# %%
#model = MobileNetV2(input_shape=(32, 32, 1), include_top=True, weights=None, classes=2)

# %%
#model.summary()

# %%


# %%
