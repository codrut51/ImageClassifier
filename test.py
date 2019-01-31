import tensorflow as tf
from tensorflow import keras
from flask_restful import Resource
from flask.json import jsonify
from flask import request


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import setup as st

#database 
import connection as conn 

data = conn.get_train_info()
classes = data['classes']

class Test(Resource):
    def get(self):
        model = tf.keras.models.load_model("models/image-7.h5")#image-2.h5
        # model = tf.keras.models.load_model("models/image-2.h5")#image-2.h5
        img1 = request.args.get('filename')
        img = self.read_test_image_file("test_imgs/" + img1)
        return jsonify({
            'prediction': str(model.predict(img)),
            'class': classes[np.argmax(model.predict(img))]
        })

    def read_test_image_file(self, file_name):
        imgs = []
        dataset = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                width_shift_range=0.01,
                height_shift_range=0.01,
                shear_range=0.2,
                zoom_range=0.4,
                horizontal_flip=True)
        
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(st.SIZE*(st.CLASSES)), int(st.SIZE)))
        # img = np.array(img)
        img = np.resize(img, st.INPUT_SHAPE)
        imgs.append(img)
        imgs.append(img)
        imgs.append(img)
        imgs.append(img)
        imgs.append(img)
        imgs.append(img)
        imgs = np.array(imgs)
        dataset.fit(imgs)
        X_batch = dataset.flow(imgs, imgs, batch_size=6, shuffle=False)

        imgs1 = []
        for inps, outs in X_batch:
            for inp in inps:
                imgs1.append(inp)
                break
            break
        # imgs1 = np.expand_dims(imgs1, axis=1)
        imgs1 = np.array(imgs1)
        return imgs1


# model = tf.keras.models.load_model("models/image-7.h5")#image-2.h5
# # model = tf.keras.models.load_model("models/image-2.h5")#image-2.h5
# img = read_test_image_file("test_imgs/test.jpg")
# img1 = read_test_image_file("test_imgs/test1.jpg")
# img2 = read_test_image_file("test_imgs/test2.jpg")
# img5 = read_test_image_file("test_imgs/test5.jpg")
# img6 = read_test_image_file("test_imgs/test6.jpg")
# print(classes)
# print(classes[np.argmax(model.predict(img))])
# print(model.predict(img))
# print(classes[np.argmax(model.predict(img1))])
# print(model.predict(img1))
# print(classes[np.argmax(model.predict(img2))])
# print(model.predict(img2))
# print(classes[np.argmax(model.predict(img5))])
# print(model.predict(img5))
# print(classes[np.argmax(model.predict(img6))])
# print(model.predict(img6))
