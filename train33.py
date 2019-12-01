# %%
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#from keras.applications import MobileNetV2
plt.style.use('dark_background')
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization

tb_hist = TensorBoard(log_dir='./graph3', histogram_freq=0, write_graph=True, write_images=True)
# %%
'''
# Load Dataset
'''
chanDim = -1
# %%
import glob

# %%
imgs = glob.glob('./yawning/*/*.jpg')

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
x_train = np.empty((len(train_imgs), 64, 74, 1))
y_train = np.empty(len(train_imgs))

x_valid = np.empty((len(valid_imgs), 64, 74, 1))
y_valid = np.empty(len(valid_imgs))

# %%
for idx, train in enumerate(train_imgs):
    x_train[idx] = np.expand_dims(np.array(Image.open(train).resize((74, 64), Image.BICUBIC)), -1)
    #print(train.split('/')[-1].split('_')[0])
    if(train.split('/')[-1].split('_')[0] == 'yawn'):
        y_train[idx] = 1
    else :
        y_train[idx] = 0
    #y_train[idx] = int(train.split('/')[-1].split('_')[0])

# %%
for idx, valid in enumerate(valid_imgs):
    x_valid[idx] = np.expand_dims(np.array(Image.open(valid).resize((74, 64), Image.BICUBIC)), -1)
    if(train.split('/')[-1].split('_')[0] == 'yawn'):
        y_train[idx] = 1
    else :
        y_train[idx] = 0

# %%

print("**************shape*********")
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

# %%
'''
# Preview
'''

# %%
plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((64, 74)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_valid[4]))
plt.imshow(x_valid[4].reshape((64, 74)), cmap='gray')

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

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

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
inputs = Input(shape=(64, 74, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)
net = BatchNormalization(axis=chanDim)(net)
net = Dropout(0.25)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)
net = BatchNormalization(axis=chanDim)(net)
net= Dropout(0.25)(net)


net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)
net = BatchNormalization(axis=chanDim)(net)
net = Dropout(0.25)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = BatchNormalization()(net)
net = Dropout(0.5)(net)


#classifier
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

hist = model.fit_generator(
    train_generator, epochs=100, validation_data=valid_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='valid_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='valid_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05), tb_hist
    ]
)


model.save('models/train_yawn_100.h5')

# %%
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# %%
#start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

#model.fit_generator(
#    train_generator, epochs=50, validation_data=valid_generator,
#    callbacks=[
#        ModelCheckpoint('models/%s.h5' % (start_time), monitor='valid_acc', save_best_only=True, mode='max', verbose=1),
#        ReduceLROnPlateau(monitor='valid_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
##    ]
##)

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
