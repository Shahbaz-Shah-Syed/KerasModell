# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:07:13 2019

@author: Shahbaz Shah Syed
"""

#load and evaluate a saved model
#from numpy import loadtxt
from keras.models import load_model
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, roc_auc_score

#load model
model=load_model('RezyklatModel.h5')
#summarize model
model.summary()

#load dataset

Test2 = 'Rezyklat_Test_Set.xlsx'
Tabelle_Test_Set = pd.read_excel(Test2,names=['0','1',
                                '2','3',
                                '4','5',
                                '6','7',
                                '8',
                                '9', 'Gewicht_Value'])


#input Data
x_test = Tabelle_Test_Set.drop(columns=['Gewicht_Value']) #x values, inputsrt

#Extract output value column
y_test = Tabelle_Test_Set['Gewicht_Value']

#evaluate the model
# =============================================================================
# score = model.evaluate(x_test,y_test, verbose = 0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# =============================================================================


#Do a Prediction and check the Precision
y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded,dtype='int64')
tn, fp, fn, tp = confusion_matrix(y_test,y_pred1).ravel()
print('\n')
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)
print("Total True: ", (tp+tn))
print("Total False: ", (fp+fn))
#print(ConfusionMatrix(y_test,y_pred1))
print('\n')
precision = precision_score(y_test,y_pred1)
print('The Test Precision Score is: %.2f%%' % (precision*100))
recall = recall_score(y_test,y_pred1)
print('The Recall Score is: %.2f%%'% (recall*100))
accuracy = accuracy_score(y_test,y_pred1)
print('The Test Accuracy Score is: %.2f%%' % (accuracy*100))
f1 = f1_score(y_test,y_pred1)
print('The Test F1 Score is: %.2f%%' % (f1*100))
jaccard = jaccard_score(y_test,y_pred1)
print('The Test Jaccard Score Score is: %.2f%%' % (jaccard*100))
rocauc = roc_auc_score(y_test,y_pred1)
print('The Test ROC AUC Score is: %.2f%%' % (rocauc*100))
