import os
import pandas as pd
import numpy as np
import time
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
sys.stderr = stderr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#_____________________________________________________________________________#

#ANN parameters
    
params = {
    "epochs": 100,
    'l_r': 0.00010000,
    }    

Nodes = 32

#_____________________________________________________________________________# 

#Definimos estrctura de la red#
def build_model():
    
    model = Sequential()
    model.add(Dense(Nodes, activation = 'linear', input_shape=(3,)))
    model.add(Dropout(0.2))
    model.add(Dense(Nodes, activation = 'linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) 
    
    return model


def model_fit_pretrained(model_name, x_test_t, y_test_t):
    
    class_model = load_model(os.path.join(model_path, model_name))
    y_pred = class_model.predict(x_test_t)
    error = mean_squared_error(y_test_t, y_pred, multioutput='raw_values')
    
    return y_pred, error


def model_fit_t(model_name, x_train, y_train, x_val, y_val, x_test_t, y_test_t):
    
    class_model = build_model()
    
    
    mcheckp = ModelCheckpoint(os.path.join(model_path, model_name), 
                              monitor='val_mean_absolute_error', verbose=0,
                              save_best_only=True, 
                              save_weights_only=False,
                              mode='auto', period=1)

    tlogger = CSVLogger(os.path.join(log_path, log_filename), separator = ',', append=False)
    
    history = class_model.fit(x_train, y_train, epochs=params["epochs"], 
                              verbose = 0, shuffle = True, 
                              validation_data = (x_val, y_val), batch_size = 100,
                              callbacks = [mcheckp, tlogger])
    
    y_pred = class_model.predict(x_test_t)
    error = mean_squared_error(y_test_t, y_pred, multioutput='raw_values')
    
    return y_pred, error, history

#_____________________________________________________________________________# 

figure_path = r'C:\Users\jaime\Documents\Python\TFM\Figures'
data_path = r'C:\Users\jaime\Documents\Python\TFM\Data'
log_path = r'C:\Users\jaime\Documents\Python\TFM\Logs'
model_path = r'C:\Users\jaime\Documents\Python\TFM\Models'
timestr = time.strftime("%d-%m @ %H.%M_%S")

#_____________________________________________________________________________# 

model_load = input('    Do you want to load a pretrained model?:  Y/N - ')


if model_load == 'Y':
    load_state = True
else:
    load_state = False
    
model_name = input('    Insert a name for the model file (must be the same name for the three models, changing only the end): ')
model_nameW = model_name + 'W' + 'hdf5'
model_nameX = model_name + 'X' + 'hdf5'
model_nameY = model_name + 'Y' + 'hdf5'
model_nameZ = model_name + 'Z' + 'hdf5' 
data_name = input('    Insert the name of the data file to use: ')
data_name = data_name + '.csv'
log_filename = "tlog_"+ timestr + ".csv"
figureW_n = "W_" + timestr +  ".png"
figureX_n = "X_" + timestr +  ".png"
figureY_n = "Y_" + timestr +  ".png"
figureZ_n = "Z_" + timestr +  ".png"
loss_f_nW = "loss_W_" + timestr + ".png"
loss_f_nX = "loss_X_" + timestr + ".png"
loss_f_nY = "loss_Y_" + timestr + ".png"
loss_f_nZ = "loss_Z_" + timestr + ".png"

#_____________________________________________________________________________# 

#Data read from CSV file

data = pd.read_csv(os.path.join(data_path, data_name),engine='python', nrows = 29000)
train_cols = ["A1","A2","A3"]
target_cols = ['IW','IX', 'IY', 'IZ']

data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
print("    Train and Test size", len(data_train), len(data_test))

x = data_train.loc[:,train_cols].values
mean_x = x.mean(axis=0)
x -= mean_x
std_x = x.std(axis=0)
x_train = x/std_x
x_test = data_test.loc[:,train_cols].values
x_test -= mean_x
x_test /= std_x
x_val, x_test_t = np.array_split(x_test, 2)

y_train = data_train.loc[:,target_cols].values
y_test = data_test.loc[:,target_cols].values

y_val, y_test_t = np.array_split(y_test, 2)

#ANN Model building or loading

y_test_tW = y_test_t[:,0]
y_test_tX = y_test_t[:,1]
y_test_tY = y_test_t[:,2]
y_test_tZ = y_test_t[:,3]
y_trainW = y_train[:,0]
y_trainX = y_train[:,1]
y_trainY = y_train[:,2]
y_trainZ = y_train[:,3]
y_valW = y_val[:,0]
y_valX = y_val[:,1]
y_valY = y_val[:,2]
y_valZ = y_val[:,3]


   
if load_state:

    (y_predW, errorW) = model_fit_pretrained(model_nameW, x_test_t, y_test_tW)
    (y_predX, errorX) = model_fit_pretrained(model_nameX, x_test_t, y_test_tX)
    (y_predY, errorY) = model_fit_pretrained(model_nameY, x_test_t, y_test_tY)
    (y_predZ, errorZ) = model_fit_pretrained(model_nameZ, x_test_t, y_test_tZ)
    
else:
 
    (y_predW, errorW, historyW) = model_fit_t(model_nameW, x_train, y_trainW, x_val, y_valW, x_test_t, y_test_tW)
    (y_predX, errorX, historyX) = model_fit_t(model_nameX, x_train, y_trainX, x_val, y_valX, x_test_t, y_test_tX)
    (y_predY, errorY, historyY) = model_fit_t(model_nameY, x_train, y_trainY, x_val, y_valY, x_test_t, y_test_tY)
    (y_predZ, errorZ, historyZ) = model_fit_t(model_nameZ, x_train, y_trainZ, x_val, y_valZ, x_test_t, y_test_tZ)
        


print("    Error is", errorW, errorX, errorY, errorZ)

#Plotting

#save_state = input('    Do you want to save the figures?:  Y/N  -  ')
save_state = 'Y'

if save_state == 'Y':
    saving = True
    state = 'Yes'
    
elif save_state == 'N':
    saving = False
    state = 'No'
    
else:
    print('    Incorrect input, the default selection is No')

# Visualize the training data
if load_state != True:
    plt.figure()
    plt.plot(historyW.history['val_mean_absolute_error'], '-g')
    plt.title('MAE Absolute Error W')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    if saving: 
        plt.savefig(os.path.join(figure_path, loss_f_nW))  
    plt.figure()
    plt.plot(historyX.history['val_mean_absolute_error'], '-b')
    plt.title('Mean Absolute Error X')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    if saving: 
        plt.savefig(os.path.join(figure_path, loss_f_nX))
    plt.figure()
    plt.plot(historyY.history['val_mean_absolute_error'], '-r')
    plt.title('Mean Absolute Error Y')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    if saving: 
        plt.savefig(os.path.join(figure_path, loss_f_nY))
    plt.figure()
    plt.plot(historyZ.history['val_mean_absolute_error'], '-k')
    plt.title('Mean Absolute Error Z')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    if saving: 
        plt.savefig(os.path.join(figure_path, loss_f_nZ))

# Visualize the prediction
plt.figure()
plt.plot(y_predW)
plt.plot(y_test_tW)
plt.title('Prediction vs Real Value')
plt.ylabel('W')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureW_n))
        
plt.figure()        
plt.plot(y_predX)
plt.plot(y_test_tX)
plt.title('Prediction vs Real Value')
plt.ylabel('X')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureX_n))

plt.figure()
plt.plot(y_predY)
plt.plot(y_test_tY)
plt.title('Prediction vs Real Value')
plt.ylabel('Y')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureY_n))

plt.figure()
plt.plot(y_predZ)
plt.plot(y_test_tZ)
plt.title('Prediction vs Real Value')
plt.ylabel('Z')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureZ_n))

