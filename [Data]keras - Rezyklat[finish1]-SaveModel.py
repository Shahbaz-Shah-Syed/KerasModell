# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:16:33 2019

@author: Shahbaz Shah Syed
"""

###BINARY CLASSIFICATION PROBLEM
import timeit
#start = timeit.default_timer()

#Import the required Libraries
#from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, jaccard_score, classification_report, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
#from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
#from keras.utils import to_categorical
#from random import seed
from keras.optimizers import Adam




##EXTRACT THE DATA AND SPLIT INTO TRAINING, VALIDATION AND TESTING------------


Test2 = 'Rezyklat_TEST_DATA.xlsx'
Tabelle = pd.read_excel(Test2,names=['Plastzeit Z [s]','Massepolster [mm]',
                                'Zylind. Z11 [°C]','Entformen[s]',
                                'Nachdr Zeit [s]','APC+ Vol. [cm³]',
                                'Energie HptAntr [Wh]','Fläche WkzDr1 [bar*s]',
                                'Fläche Massedr [bar*s]',
                                'Fläche Spritzweg [mm*s]', 'Gewicht'])


#SHUFFLING The Data before splitting, so that each set has some Data from all categories/classes
#Tabelle.reindex(np.random.permutation(Tabelle.index))
#seed(4)
Tabelle = Tabelle.sample(frac=1, random_state=4).reset_index(drop=True)
# =============================================================================
# 
# =============================================================================
# =============================================================================
# for i in range(4):
#     print(Tabelle[i])
# 
# =============================================================================

#input Data
Tabelle_feat = Tabelle.drop(columns=['Gewicht']) #x values, inputsrt

#Extract output value column
Gewicht = Tabelle['Gewicht']


#Toleranz festlegen
toleranz = 1

#guter Bereich für Gewicht
Gewicht_mittel = Gewicht.mean()
Gewicht_abw = Gewicht.std()
Gewicht_tol = Gewicht_abw*toleranz

Gewicht_OG = Gewicht_mittel+Gewicht_tol
Gewicht_UG = Gewicht_mittel-Gewicht_tol


#Gewicht Werte in Gut und Schlecht zuordnen
G = []
for element in Gewicht:
    if element > Gewicht_OG or element < Gewicht_UG:
        G.append(0)
    else:
        G.append(1)      
G = pd.DataFrame(G)
G=G.rename(columns={0:'Gewicht_Value'})
Gewicht = pd.concat([Gewicht, G], axis=1)

#Target Data in 1s and 0s
Gewicht_Value = Gewicht['Gewicht_Value'] #y values, mixed 1 and 0

scaler = StandardScaler()
scaler.fit(Tabelle_feat)
#transform can only work if table has been fitted before
scaled_features = scaler.transform(Tabelle_feat)

Tabelle_feat_scaled = pd.DataFrame(scaled_features)


#Split the train, validation and test set
Tabelle_feat=Tabelle.shape[0]
Train_rows = int(Tabelle_feat*0.8)
Val_Test_rows = int(Tabelle_feat*0.1)

# =============================================================================
# x_train = Tabelle_feat_scaled[:Train_rows]
# x_val = Tabelle_feat_scaled[Train_rows:(Train_rows+Val_Test_rows)]
# x_test = Tabelle_feat_scaled[(Train_rows+Val_Test_rows):(Train_rows+Val_Test_rows*2)]
# 
# y_train = Gewicht_Value[:Train_rows]
# y_val = Gewicht_Value[Train_rows:(Train_rows+Val_Test_rows)]
# y_test = Gewicht_Value[(Train_rows+Val_Test_rows):(Train_rows+Val_Test_rows*2)]
# =============================================================================

X = Tabelle_feat_scaled[:(Train_rows+Val_Test_rows)]
y = Gewicht_Value[:(Train_rows+Val_Test_rows)]

x_test = Tabelle_feat_scaled[(Train_rows+Val_Test_rows):]
y_test = Gewicht_Value[(Train_rows+Val_Test_rows):]

test_set = pd.concat([x_test, y_test], axis = 1)

test_set.to_excel("Rezyklat_Test_Set.xlsx", index = False)


num_epochs = 175
hidden_units = 512
num_batchsize = 32
#learningrate = 0.001
kernel = 0.001
dropout_value = 0.2


#Create a Neural Network
def build_model():
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(10,), kernel_regularizer=l2(kernel))) #
    model.add(Dropout(dropout_value, noise_shape=None, seed=1))
    model.add(Dense(hidden_units, activation='relu', kernel_regularizer=l2(kernel))) #
# =============================================================================
#     model.add(Dropout(dropout_value, noise_shape=None, seed=1))
#     model.add(Dense(hidden_units, activation='relu'))
# =============================================================================
    model.add(Dropout(dropout_value, noise_shape=None, seed=1))
    model.add(Dense(1, activation='sigmoid'))

    #Compile the Model/Neural Network
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #Adam(learning_rate = learningrate)

    #Check the Model summary
    #model.summary()
    return model



#K-FOLD CROSS-VALIDATION, händisch ohne scikit

k=9
num_val_samples = len(X) // k

all_scores=[]
all_mae_histories = []
val_split = 1/9

train_acc = []
train_loss = []
val_acc = []
val_loss = []

for turn in range(1):
    for i in range(k):
        print('processing fold #', i)
        val_data = X[i * num_val_samples: (i+1)*num_val_samples]
        val_targets = y[i*num_val_samples: (i+1)*num_val_samples]
        
        partial_train_data = np.concatenate([X[:i*num_val_samples], X[(i+1)*num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([y[:i*num_val_samples], y[(i+1)*num_val_samples:]], axis = 0)
        
        #TRAIN NEURAL NETWORK
        #es = EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = 1)
        
        model = build_model()
        model_output = model.fit(partial_train_data, partial_train_targets, validation_data =(val_data, val_targets), epochs=num_epochs, batch_size = num_batchsize, verbose = 0) #, callbacks = [es]
        
        #DOCUMENT NEURAL NETWORK OUTCOMES    
        #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        #all_scores.append(val_mae)
        #mae_history = np.mean(model_output.history['val_accuracy'])
        #all_mae_histories.append(mae_history)
    
    # =============================================================================
    # average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    # 
    # plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation MAE')
    # plt.show()
    # =============================================================================
    
    # =============================================================================
    # model_output = model.fit(x_train, y_train, epochs = 20, batch_size=20, validation_data =(x_val, y_val))
    # =============================================================================
    
    tr_loss = np.mean(model_output.history['loss'])
    tr_acc = np.mean(model_output.history['accuracy'])
    v_acc = np.mean(model_output.history['val_accuracy'])
    v_loss = np.mean(model_output.history['val_loss'])
    
    print('Training Accuracy : ' , np.mean(model_output.history['accuracy']))
    print('Training Loss : ' , np.mean(model_output.history['loss']))
    print('Validation Accuracy : ' , np.mean(model_output.history['val_accuracy']))
    print('Validation Loss : ' , np.mean(model_output.history['val_loss']))
    
    train_acc.append(tr_acc)
    train_loss.append(tr_loss)
    val_acc.append(v_acc)
    val_loss.append(v_loss)
    
    #Plot the model accuracy over epochs
    # Plot training & validation accuracy values
    plt.plot(model_output.history['accuracy'])
    plt.plot(model_output.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    #plt.savefig("accuracy"+str(turn)+".png")
    
    
    # Plot training & validation loss values
    plt.plot(model_output.history['loss'])
    plt.plot(model_output.history['val_loss'])
    plt.title('model_output loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    #plt.savefig("loss"+str(turn)+".png")

stop = timeit.default_timer()
print('Time: ', stop)  

model.save("RezyklatModel.h5")
print("Saved Model to disk")

