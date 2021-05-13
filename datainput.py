#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
from keras.utils import np_utils
import tensorflow as tf
import cv2
from netmodel import *
from flask import *



app = Flask(__name__)

@app.route('/Check/<filepath>', methods=['GET'])
def Check(filepath):

    test_pred = model.predict(val_x)
    test_pred = np.argmax(test_pred,1)
    return test_pred

def trainingmodel():
    train = pd.read_csv("digit-recognizer/train.csv")
    test = pd.read_csv('digit-recognizer/test.csv')
    train_y = train['label']
    train_x = train.drop('label',axis=1)
    train_x = train_x.astype("float32")/255.0
    train_x = train_x.values.reshape(-1,28,28,1)
    test_x = test.astype("float32")/255.0
    test_x = test_x.values.reshape(-1,28,28,1)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1) # split into training and validation 10%
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,15))
    rows = 3
    columns = 3
    for i in range(9):
        fig.add_subplot(rows, columns, i+1)
        img_arr = test.iloc[i,1:].values.reshape(28,28)
        plt.imshow(img_arr)
        plt.axis('off')
        plt.title("Label" + ' ' + str(test.iloc[i,0]))
    """
    train_y = np_utils.to_categorical(train_y,10)
    model = modelmaker2()
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    model.fit(train_x,train_y,batch_size=64,epochs=2,verbose=1)


if __name__ == '__main__':
    train = False
    if train == True:
        trainingmodel()
    else:
        json_file = open("model1.json", 'r')
        lmodel = json_file.read()
        json_file.close()
        loaded_model = model_from_json(lmodel)
        loaded_model.load_weights("model.h5")
        loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    app.run(port=8081, debug=True)
