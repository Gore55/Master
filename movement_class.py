import numpy as np
import pandas as pd
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
sys.stderr = stderr
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time

#_____________________________________________________________________________#

params = {
    "epochs": 200,
    "l_r": 0.00010000,
}

#_____________________________________________________________________________#

def build_model():
    
    model = Sequential()
    model.add(Dense(50, activation = 'relu', input_shape=(4,)))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(5, activation = 'softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    return model

def model_fit_t(model_name, x_train, y_train, x_val, y_val, x_test_t, y_test_t):
    
    class_model = build_model()
    
    mcheckp = ModelCheckpoint(os.path.join(model_path, model_name), 
                              monitor='acc', verbose=1,
                              save_best_only=True, 
                              save_weights_only=False,
                              mode='auto', period=1)

    tlogger = CSVLogger(os.path.join(log_path, log_filename), separator = ',', append=False)
    
    history = class_model.fit(x_train, y_train, epochs=params["epochs"], 
                              verbose = 1, shuffle = True, 
                              validation_data = (x_val, y_val), batch_size = 50,
                              callbacks = [mcheckp, tlogger])
    
    results = class_model.evaluate(x_test_t, y_test_t)
    
    
    return results, history

def model_fit_pretrained(model_name, x_test_t, y_test_t):
    
    class_model = load_model(os.path.join(model_path, model_name))
    results = class_model.evaluate(x_test_t, y_test_t)    
    return results

def one_hot_targets(y):
    he_y = np.zeros((y.shape[0],5))
    for i in range(0,y.shape[0]):
        if y[i] == 'Abduction':
            he_y[i,0] = 1
        elif y[i] == 'Flexion':
            he_y[i,1] = 1            
        elif y[i] == 'Horizontal_Adduction':
            he_y[i,2] = 1
        elif y[i] == 'Opening_Drill':
            he_y[i,3] = 1
        elif y[i] == 'Closing_Drill':
            he_y[i,4] = 1
    return he_y

#_____________________________________________________________________________#

figure_path = r'C:\Users\jaime\Documents\Python\TFM\Figures'
data_path = r'C:\Users\jaime\Documents\Python\TFM\Data'
log_path = r'C:\Users\jaime\Documents\Python\TFM\Logs'
model_path = r'C:\Users\jaime\Documents\Python\TFM\Models'
saving = False
Input = False
state = 'No'
timestr = time.strftime("%d-%m @ %H.%M_%S")

#_____________________________________________________________________________# 

model_load = input('    Do you want to load a pretrained model?:  Y/N - ')

if model_load == 'Y':
    load_state = True
else:
    load_state = False
    
model_name = input('    Insert a name for the model file: ')
model_name = model_name + 'hdf5'
data_name = input('    Insert the name of the data file to use: ')
data_name = data_name + '.csv'
log_filename = "tlog_"+ timestr + ".csv"
figure_n = "acc_class_" + timestr +  ".png"
loss_f_n = "loss_class_" + timestr + ".png"

#_____________________________________________________________________________#

#Data read from CSV file

data = pd.read_csv(os.path.join(data_path, data_name),engine='python')
train_cols = ["A1","A2","A3","A4"]
target_cols = ["Movement"]
data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
x = data_train.loc[:,train_cols].values


normi = Normalizer().fit(x)
x_train = normi.fit_transform(x)
x_test = normi.transform(data_test.loc[:,train_cols])
x_val, x_test_t = np.array_split(x_test, 2)

y_train = data_train.loc[:,target_cols].values
y_test = data_test.loc[:,target_cols].values
y_train = one_hot_targets(y_train)
y_test = one_hot_targets(y_test)
y_val, y_test_t = np.array_split(y_test, 2)


if load_state:
    
    (results) = model_fit_pretrained(model_name, x_test_t, y_test_t)
    
else:

    (results, history) = model_fit_t(model_name, x_train, y_train, x_val, y_val, x_test_t, y_test_t)

print(results)

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(figure_path, figure_n))

plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.savefig(os.path.join(figure_path, loss_f_n))  

    
