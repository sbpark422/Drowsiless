# %%
# import the needed libraries
import tensorflow as tf
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.python.keras.models import load_model

# path we wanna save our converted TF-model
#MODEL_PATH = "./model/tensorflow/big/model1"
MODEL_PATH = "./models"

# load the Keras model
#model = load_model('./model/modelLeNet5.h5')
model = load_model('./models/train_mrl_50.h5')

# save the model to Tensorflow model
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, MODEL_PATH)

print("Keras model is successfully converted to TF model in "+MODEL_PATH)

# %%
