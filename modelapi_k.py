import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import keras
from PIL import Image
from label_list_store_k import base_list
from label_list_store_k import matra_list
from label_list_store_k import dot_list
from label_list_store_k import base_model
from label_list_store_k import matra_model
from label_list_store_k import dot_model


print('\n\n=======================IMPORTANT=======================\n')
print('The images with class labels 2306, 2416, and 2362 actually all look the same in the train dataset. These have all been relabelled to 2362.')
print('Hence a prediction of 2306 made by this modelapi should be considered equivalent to a prediction of any of (2306, 2416, and 2362) when trying to compare this model with a model that did not make this assumption.')
print('\n=======================================================\n\n')

def pad_resize(img):
    top = int((224 - img.shape[0])/2)
    left = int((224 - img.shape[1])/2)
    bottom = 224 - img.shape[0] - top
    right = 224 - img.shape[1] - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    img = img/255.
    img = cv2.resize(img, (64,64)) 
    return img

def preprocess(filename):
    #plt.imsave("temp.png",img)
    img = cv2.imread(os.path.join('',filename),0)
    blur = cv2.GaussianBlur(img,(9,9),0)# KADD
    #img = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
    blur = cv2.GaussianBlur(img,(9,9),0)
    a,img = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = pad_resize(img)
    #plt.imshow(img,'gray')
    #plt.show()
    return img

def predict(filename):

    img = preprocess(filename)
    X_test = np.reshape(img,(-1,64,64,1))


    base_pred = base_list[np.argmax(base_model.predict(X_test))]
    matra_pred = matra_list[np.argmax(matra_model.predict(X_test))]
    dot_pred = dot_list[np.argmax(dot_model.predict(X_test))]

    total_pred = [base_pred,matra_pred,dot_pred]

    result = [i for i in total_pred if i!=0]
    
    #print("Predicted Integers: " + str(total_pred))
    #print("Predictions in Hex: " + str([hex(i) for i in total_pred]))
    return result

def perf_measure(y_actual, y_hat):
    y_actual = set(y_actual)
    y_pred = set(y_hat)
    tp = y_actual.intersection(y_pred)
    fn = y_actual.difference(y_pred)
    fp = y_pred.difference(y_actual)
    return(len(tp), len(fn), len(fp))

if __name__ == '__main__':
    from label_list_store_k import tests
    from sklearn.metrics import confusion_matrix
    score = 0.0
    for i in tests:
        #print(i)
        original = i[:-4].split('_')[3:]
        original = [i for i in original if int(i) > 1000]
        prediction = [str(j) for j in predict(i)]
        tp, fn, fp = perf_measure(original, prediction)
        score += (float(tp) / float(tp + fp + fn))
        print('The prediction on {} is: {}, with score {}'.format(original, prediction, score))
    print('\nfinal score: {}, on {} total images, normalised to {}'.format(score, len(tests), score/len(tests)))


