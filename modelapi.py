import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import keras
from keras.models import load_model

base_list = [2310, 2311, 2312, 2313, 2315, 2319, 2325, 2327, 2328, 2330, 2331,
       2332, 2334, 2335, 2336, 2337, 2338, 2340, 2341, 2342, 2343, 2344,
       2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2357, 2358,
       2359, 2360, 2361, 2384, 2392, 2399, 2404, 2405, 2406, 2407, 2408,
       2409, 2410, 2411, 2412, 2413, 2414, 2415, 2424, 2429]
matra_list = [0, 2306, 2362, 2363, 2364, 2366, 2367, 2368, 2369, 2370, 2372,
       2375, 2376, 2379, 2380, 2382, 2387, 2390]
dot_list = [0, 1]

def pad_resize(img):
    top = int((224 - img.shape[0])/2)
    left = int((224 - img.shape[1])/2)
    bottom = 224 - img.shape[0] - top
    right = 224 - img.shape[1] - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    img = img/255.
    img = cv2.resize(img, (64,64)) #KADD
    return img

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    blur = cv2.GaussianBlur(img,(9,9),0)
    a,img = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = pad_resize(img)


def predict(img):
    img = preprocess(img)
    X = np.reshape(img, (-1, 64*64))
    X_test = np.reshape(img,(-1,64,64,1))

    base_model = load_model('base_model.h5')
    matra_model = load_model('matra_model.h5')
    dot_model = load_model('dot_model.h5')

    base_pred = base_list[np.argmax(model.predict(X_test))]
    matra_pred = matra_list[np.argmax(model.predict(X_test))]
    dot_pred = dot_list[np.argmax(model.predict(X_test))]

    total_pred = [base_pred,matra_pred,dot_pred]

    total_pred = filter(lambda a : a != 0, total_pred)
    return total_pred

im = plt.imread('ok.png')
print(predict(im))

