
#%%
#%matplotlib inline

#%%
import tensorflow as tf

#%%
sess = tf.InteractiveSession()

#%%
import keras.backend as K
K.set_image_data_format("channels_first")

import keras
import numpy as np

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

#%%
# Used to save and load training histories
import pickle
from collections import defaultdict

import resource, sys

# limit recursion depth
limit = sys.getrecursionlimit()
print(limit)

#resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
#sys.setrecusionlimit(2**29 - 1)

#%%
from scipy import io as spio
emnist = spio.loadmat("/Users/andereggt/datasets/emnist/matlab/emnist-digits.mat")

#%%
# Load training dataset and labels
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

y_train = emnist["dataset"][0][0][0][0][0][1]


#%%
# Load test dataset and labels
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

y_test = emnist["dataset"][0][0][1][0][0][1]


#%%
# Store labels for visualization
train_labels = y_train
test_labels = y_test

#%%
print("Training data shape: ", x_train.shape)
print("Training label shape: ", y_train.shape)

#%%
# Normalize datasets
x_train /= 255
x_test /= 255
print(x_train)

#%%
# Reshape using matlab order
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

print("Reshaped training data: ", x_train.shape)


#%%
# labels should be onehot encoded
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("One-hot encoded label shape: ", y_train.shape)

#%%
# Verify data has been loaded correctly
samplenum = 5437

import matplotlib.pyplot as plt
img = x_train[samplenum]
plt.imshow(img[0], cmap='gray')

#%%
# Reshape test labels
test_labels = test_labels.reshape(40000)
print("Reshaped test labels: ", test_labels.shape)

#%%
# Define Model
# Calculate mean and standard deviation
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

# Define function to normalize input data
def norm_input(x): return (x-mean_px)/std_px

# Batchnorm + dropout + data augmentation
def create_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28), output_shape=(1,28,28)),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Conv2D(64, (3,3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(64, (3,3)),
        LeakyReLU(),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#%%
# Use Keras data augmentation
batch_size = 512

from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                         height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')
batches = gen.flow(x_train, y_train, batch_size=batch_size)
test_batches = gen.flow(x_test, y_test, batch_size=batch_size)
steps_per_epoch = int(np.ceil(batches.n/batch_size))
validation_steps = int(np.ceil(test_batches.n/batch_size))

#%%
# Visualize data gen
import matplotlib.pyplot as plt
img = x_train[1]
plt.imshow(img[0], cmap='gray')

#%%
# Get augmented images
img = np.expand_dims(img, axis=0)
aug_iter = gen.flow(img)

aug_img = next(aug_iter)[0].astype(np.float32)
print("Augmented image shape: ", aug_img.shape)

import matplotlib.pyplot as plt

f = plt.figure(figsize=(12,6))
for i in range(8):
    sp = f.add_subplot(2, 26//3, i+1)
    sp.axis('Off')
    aug_img = next(aug_iter)[0].astype(np.float32)
    plt.imshow(aug_img[0], cmap='gray')


#%%
# Create 10 models
models = []
weights_epoch = 0

for i in range(10):
    m = create_model()
    models.append(m)

#%%
eval_batch_size = 512
num_iterations = 1
num_epochs = 10

import os
if not os.path.exists("dropout_0.2"):
    os.mkdir("dropout_0.2")
if not os.path.exists("dropout_0.2/weights"):
    os.mkdir("dropout_0.2/weights")
if not os.path.exists("dropout_0.2/history"):
    os.mkdir("dropout_0.2/history")

for iteration in range(num_iterations):
    cur_epoch = (iteration + 1) * num_epochs + weights_epoch

    for i, m in enumerate(models):
        m.optimizer.lr = 0.000001
        h = m.fit_generator(batches, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=0,
                            validation_data=test_batches, validation_steps=validation_steps)
        m.save_weights("dropout_0.2/weights/{:03d}epochs_weights_model_{}.pkl".format(cur_epoch, i))
    
    # evaluate test error rate for ensemble
    all_preds = np.stack([m.predict(x_test, batch_size=eval_batch_size) for m in models])
    avg_preds = all_preds.mean(axis=0)
    test_error_ensemble = (1 - keras.metrics.categorical_accuracy(y_test, avg_preds).eval().mean()) * 100

    # write test error rate for ensemble and every single model to text file
    with open("dropout_0.2/history/test_errors_epoch_{:03d}.txt".format(cur_epoch), "w") as text_file:
        text_file.write("epoch: {} test error on ensemble: {}\n".format(cur_epoch, test_error_ensemble))
        
        for m in models:
            pred = np.array(m.predict(x_test, batch_size=eval_batch_size))
            test_err = (1 - keras.metrics.categorical_accuracy(y_test, pred).eval().mean()) * 100
            text_file.write("{}\n".format(test_err))

#%%
eval_batch_size = 512
all_preds = np.stack([m.predict(x_test, batch_size=eval_batch_size) for m in models])
avg_preds = all_preds.mean(axis=0)
print("Ensemble error rate: ", (1 - keras.metrics.categorical_accuracy(y_test, avg_preds).eval().mean()) * 100)

#%%
