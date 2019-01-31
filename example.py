import pickle as pk
import numpy as np
import tensorflow as tf 
import tensorboard
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam, SGD
import setup as st
from keras.preprocessing import image
import pickle
import datetime

from sklearn.utils import shuffle

pickle_in = open("./data/X.pickle", "rb")
train_imgs = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("./data/Y.pickle", "rb")
train_labels = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("./data/X_test.pickle", "rb")
test_imgs = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("./data/Y_test.pickle", "rb")
test_labels = pickle.load(pickle_in)
pickle_in.close()
print(train_imgs.shape)
train_imgs = np.array(train_imgs)
test_imgs = np.array(test_imgs)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)



print(test_imgs.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=train_imgs.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
# model.add(Dropout(0.4))
model.add(Dense(4, activation="softmax"))

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,#'sparse_categorical_crossentropy'
              optimizer=Adam(lr=1e-3, decay=1e-6),
              metrics=['accuracy'])

data = str(datetime.datetime.now()).split(":")
result = ""
for p in (range(len(data))):
    result += "." + data[p]


tbCallBack = keras.callbacks.TensorBoard(log_dir='./logdir/{0}'.format(result), histogram_freq=0, write_graph=True, write_images=True)
batch_size = 30
epochs = 10

model.fit(train_imgs, train_labels, 
          batch_size= batch_size, 
          epochs=epochs, 
          validation_split=0.3, 
          validation_data=(test_imgs, test_labels),
          callbacks=[tbCallBack],
          shuffle=True,
          verbose=1)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss=tf.keras.losses.categorical_crossentropy,#'sparse_categorical_crossentropy'
                                    metrics=['accuracy'])
                                    
model.fit(train_imgs, train_labels, 
          batch_size= batch_size, 
          epochs=epochs, 
          validation_split=0.3, 
          validation_data=(test_imgs, test_labels),
          callbacks=[tbCallBack],
          shuffle=False,
          verbose=1)
          

test_loss, test_acc = model.evaluate(test_imgs, 
                                    test_labels)

print('Test accuracy: {0}, Test lost: {1}'.format(test_acc, test_loss))


model.save("models/image-127.h5")


# test_acc = accuracy.mean()
# test_loss = accuracy.var()


# test_imgs, test_labels = shuffle(test_imgs, test_labels)
# train_imgs, train_labels = shuffle(train_imgs, train_labels)s

# model = Sequential()

# model.add(CuDNNLSTM(256, input_shape=(setup.SIZE, setup.SIZE), return_sequences=True))#128
# model.add(Dropout(0.2))

# model.add(CuDNNLSTM(128))
# # model.add(Dropout(0.2))
# model.add(Input(shape=st.INPUT_SHAPE))

# model.add(Flatten())

# model.add(Dense(64, activation="relu"))
# # model.add(Dropout(0.2))

# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))

# model.add(Dense(64, activation="relu"))
# # model.add(Dropout(0.2))

# model.add(Dense(128, activation="relu"))
# # model.add(Dropout(0.2))

# model.add(Dense(6, activation="softmax"))

# dataset = image.ImageDataGenerator(
#     rotation_range=10.,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.,
#     zoom_range=1.,
#     horizontal_flip=True,
#     vertical_flip=True)

# # train_data = dataset.flow(train_imgs, train_labels, batch_size=1, shuffle=False)

# # model.fit_generator(train_data
# #  , steps_per_epoch=train_data.n/train_data.batch_size#2625#train_data.n/train_data.batch_size
# #  , epochs=10
# #  , validation_data=(test_imgs, test_labels))

# model.fit(x=train_imgs, y=train_labels, batch_size=32, epochs=15, validation_split=0.1, verbose=1, validation_data=(test_imgs, test_labels))


# dataset = keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=10.,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.,
#     zoom_range=1.,
#     horizontal_flip=True,
#     vertical_flip=True)

# train_data = dataset.flow(train_imgs, train_labels, batch_size=batch_size, shuffle=False)


# model.fit_generator((train_imgs, train_labels))      
# model.fit(test_imgs, test_labels, 
#           batch_size= batch_size, 
#           epochs=epochs, 
#           validation_split=0.3, 
#           validation_data=(train_imgs, train_labels),
#           callbacks=[tbCallBack],
#           shuffle=False,
#           verbose=1)

# model.fit_generator(train_data
#  , steps_per_epoch=150#2625#train_data.n/train_data.batch_size
#  , epochs=epochs
#  , validation_data=(test_imgs, test_labels))


# # model.fit_generator(train_data
# #  , steps_per_epoch=train_data.n/train_data.batch_size#2625#train_data.n/train_data.batch_size
# #  , epochs=10
# #  , validation_data=(test_imgs, test_labels))

# model.fit(test_imgs, test_labels, 
#           batch_size= batch_size, 
#           epochs=epochs, 
#           validation_split=0.3, 
#           validation_data=(train_imgs, train_labels),
#           callbacks=[tbCallBack],
#           shuffle=False,
#           verbose=1)

# model.fit_generator(train_data
#  , steps_per_epoch=150#2625#train_data.n/train_data.batch_size
#  , epochs=epochs
#  , validation_data=(test_imgs, test_labels))

# model.fit(x=train_imgs, y=train_labels, epochs=15, batch_size=32, validation_split=0.1, verbose=1, validation_data=(test_imgs, test_labels))
# model.fit(x=train_imgs, y=train_labels, epochs=30, batch_size=1, validation_split=0.1, verbose=1)

# def cnn_model():
#     input_layer = Input(shape=st.INPUT_SHAPE, name="input_layer")
#     use_bias = True
 
#     # Conv1
#     conv = Conv2D(32,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(input_layer)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
 
#     # Conv2
#     conv = Conv2D(32,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(activation)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
 
#     # MaxPooling1
#     max_pool = MaxPooling2D(pool_size=(2, 2))(activation)
 
#     # Conv3
#     conv = Conv2D(64,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(max_pool)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
 
#     # Conv4
#     conv = Conv2D(64,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(activation)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
 
#     # MaxPooling2
#     max_pool = MaxPooling2D()(activation)
 
#     # Conv5
#     conv = Conv2D(128,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(max_pool)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
#     # Conv6
#     conv = Conv2D(128,
#                                   kernel_size=(3, 3),
#                                   padding='same',
#                                   use_bias=use_bias,
#                                   activation=None)(activation)
#     bn = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
#     activation = Activation("relu")(bn)
 
#     # MaxPooling3
#     max_pool = MaxPooling2D()(activation)
 
#     # Dense Layer
#     flatten = Flatten()(max_pool)
 
#     # Softmax Layer

#     output = Dense(4, activation="softmax", name='output')(flatten)
 
#     return Model(inputs=input_layer, outputs=output)

# model = cnn_model()