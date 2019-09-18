import numpy as np
import pandas as pd
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
sys.stderr = stderr
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

#_____________________________________________________________________________#

#ANN parameters
    
params = {
    "batch_size": 512,
    "epochs": 200,
    "l_r": 0.00010000,
    "time_steps": 5,
    "delay": 100
    }    

TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
DELAY = params["delay"]

#_____________________________________________________________________________#   
    
def create_ts(coord_mat):
    #Matriz de entrada solo con XYZ
    d0 = coord_mat.shape[0] - TIME_STEPS - DELAY
    d1 = coord_mat.shape[1]
    x = np.zeros((d0, TIME_STEPS, d1))
    y = np.zeros((d0, d1))
    for i in range(0,d0):
        x[i] = coord_mat[i:TIME_STEPS+i]
        y[i] = coord_mat[TIME_STEPS + i + DELAY, 0:4]
    return x, y

def cut_mat(mat,batch_size):
    n_o_row = mat.shape[0]%batch_size
    if n_o_row > 0:
        return mat[:-n_o_row]
    else:
        return mat
    
def build_model():
    model = Sequential()
    model.add(LSTM(32,batch_input_shape=(BATCH_SIZE, TIME_STEPS, 4), 
                   return_sequences=True))
    model.add(LSTM(32,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(4))    
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error', metrics =['mae'])    
    return model

def model_fit_pretrained(model_name, x_test_t, y_test_t):
    
    class_model = load_model(os.path.join(model_path, model_name))
    y_pred = class_model.predict(x_test_t, batch_size=BATCH_SIZE)
    error = mean_squared_error(y_test_t, y_pred, multioutput='raw_values')
    
    return y_pred, error


def model_fit_t(model_name, x_train, y_train, x_val, y_val, x_test_t, y_test_t):
    
    recurrent_model = build_model()
    
    mcheckp = ModelCheckpoint(os.path.join(model_path, model_name), 
                          monitor='val_mean_absolute_error', verbose=1,
                          save_best_only=True, 
                          save_weights_only=False,
                          mode='auto', period=1)
    
#    plateau = ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.1, 
#                                patience=30, verbose=1, mode='auto',
#                                min_delta=0.0001, cooldown=0, min_lr=0)
    
    tlogger = CSVLogger(os.path.join(log_path, log_filename), separator = ';', append=False)
    
    history = recurrent_model.fit(x_t, y_t, epochs=params["epochs"], 
                                 verbose=1, batch_size=BATCH_SIZE,
                                 shuffle = False, validation_data = (x_val, y_val), 
                                 callbacks = [ mcheckp, tlogger])

    y_pred = recurrent_model.predict(x_test_t, batch_size=BATCH_SIZE)

    error = mean_squared_error(y_test_t, y_pred, multioutput='raw_values')
    
    return y_pred, error, history

#_____________________________________________________________________________#  
#Path and naming parameters

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
    
model_name = input('    Insert a name for the model file: ')
model_name = model_name + 'hdf5'
data_name = input('    Insert the name of the data file to use: ')
data_name = data_name + '.csv'
log_filename = "tlog_"+ timestr + ".csv"
figureW_n = "W_" + timestr +  ".png"
figureX_n = "X_" + timestr +  ".png"
figureY_n = "Y_" + timestr +  ".png"
figureZ_n = "Z_" + timestr +  ".png"
loss_f_n = "loss_" + timestr + ".png"

#_____________________________________________________________________________# 

#Data read from CSV file

data = pd.read_csv(os.path.join(data_path, data_name),engine='python', nrows = 29105)
train_cols = ["OW","OX","OY","OZ"]

#Splitting into test and training sets

data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
print("    Train and Test size", len(data_train), len(data_test))
x = data_train.loc[0:29105,train_cols].values

#Scaling and arranging Data

min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(data_test.loc[:,train_cols])

x_t, y_t = create_ts(x_train)
x_t = cut_mat(x_t, BATCH_SIZE)
y_t = cut_mat(y_t, BATCH_SIZE)

x_temp, y_temp = create_ts(x_test)
x_val, x_test_t = np.array_split(x_temp, 2)
y_val, y_test_t = np.array_split(y_temp, 2)

x_val = cut_mat(x_val, BATCH_SIZE)
x_test_t = cut_mat(x_test_t, BATCH_SIZE)
y_val = cut_mat(y_val, BATCH_SIZE)
y_test_t = cut_mat(y_test_t, BATCH_SIZE)

#RNN Model building or loading

if load_state:
    
    (y_pred, error) = model_fit_pretrained(model_name, x_test_t, y_test_t)
    
else:

    (y_pred, error, history) = model_fit_t(model_name, x_t, y_t, x_val, y_val, x_test_t, y_test_t)

print("    Error is", error)
####print("Error is", error, y_pred.shape, y_test_t.shape)

y_pred_usc = min_max_scaler.inverse_transform(y_pred)
y_test_t_usc = min_max_scaler.inverse_transform(y_test_t)

#_____________________________________________________________________________# 


# Visualize the training data
if load_state != True:
    plt.figure()
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(figure_path, loss_f_n))


# Visualize the prediction
plt.figure()
plt.plot(y_pred_usc[DELAY:y_pred_usc.shape[0],0])
plt.plot(y_test_t_usc[0:(y_test_t_usc.shape[0] - DELAY),0])
plt.title('Prediction vs Real Value')
plt.ylabel('W')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureW_n))    

plt.figure()    
plt.plot(y_pred_usc[DELAY:y_pred_usc.shape[0],1])
plt.plot(y_test_t_usc[0:(y_test_t_usc.shape[0] - DELAY),1])
plt.title('Prediction vs Real Value')
plt.ylabel('X')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureX_n))

plt.figure()
plt.plot(y_pred_usc[DELAY:y_pred_usc.shape[0],2])
plt.plot(y_test_t_usc[0:(y_test_t_usc.shape[0] - DELAY),2])
plt.title('Prediction vs Real Value')
plt.ylabel('Y')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureY_n))

plt.figure()
plt.plot(y_pred_usc[DELAY:y_pred_usc.shape[0],3])
plt.plot(y_test_t_usc[0:(y_test_t_usc.shape[0] - DELAY),3])
plt.title('Prediction vs Real Value')
plt.ylabel('Z')
plt.xlabel('Samples')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.savefig(os.path.join(figure_path, figureZ_n))
