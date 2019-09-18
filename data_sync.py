from scipy.spatial.transform import Rotation as R
import pandas as pd
import os
import numpy as np
import math

data_path = r'C:\Users\jaime\Documents\Python\TFM\Data'
final_n = 'shoulder_dataset2.csv'

def square(value):
    sq_n = value * value
    return sq_n

def data_sync(datat1,datat2):
    
    data_cols = ["Time","WE", "XE", "YE", "ZE", "W", "X", "Y", "Z", "XH", "YH", "ZH"]
    sync_cols = ['Time Stamp','A1', 'A2','A3','A4']
    qua_cols = ['qw','qx','qy','qz']
    
    t_diff = abs(float(datat1)- float(datat2))
    
    data_name = 'ott - ' + datat1 + '.csv'
    sync_name = 'EMG - ' + datat2 + '.csv'
    qua_name = 'qua - ' + datat2 + '.csv'
    
    data = pd.read_csv(os.path.join(data_path, data_name),engine='c')
    sync = pd.read_csv(os.path.join(data_path, sync_name),engine='python')
    qua = pd.read_csv(os.path.join(data_path, qua_name),engine='python')
    
    x = data.loc[:,data_cols].values
    s = sync.loc[:,sync_cols].values
    q = qua.loc[:,qua_cols].values    
    
    for j in range(1,x.shape[1]):
        for i in range(0,x.shape[0]):
            if str(x[i,j]) == 'nan':
                x[i,j] = x[i - 1,j] + (x[i,0]-x[i - 1,0])*(((x[i + 1,j])-x[i - 1,j])/(x[i + 1,0] - x[i - 1,0]))
    
    synced_data = np.zeros((s.shape[0],(16)))
    
    for j in range(0, 5):
        for i in range(0,synced_data.shape[0]):
            if j == 0:
                synced_data[i,j] = s[i,j] - s[0,0]   
            else:
                synced_data[i,j] = s[i,j]
    
    for i in range(0,synced_data.shape[0]):
#        q_t_norm = math.sqrt(square(q[i,0]) + square(q[i,1]) + square(q[i,2]) + square(q[i,3]))
#        q[i,0] /= q_t_norm
#        q[i,1] /= q_t_norm
#        q[i,2] /= q_t_norm
#        q[i,3] /= q_t_norm
#        q_t = R.from_quat([q[i,0],q[i,1],q[i,2],q[i,3]])
#        eulersitos = q_t.as_euler('xyz', degrees = True)
#        synced_data[i,5] = eulersitos[0]
#        synced_data[i,6] = eulersitos[1]
#        synced_data[i,7] = eulersitos[2]
        synced_data[i,5] = q[i,0]
        synced_data[i,6] = q[i,1]
        synced_data[i,7] = q[i,2]    
        synced_data[i,8] = q[i,3]
    
    for i in range(0,x.shape[0]):
        x[i,0] = x[i,0] - t_diff  
        
    x_t = np.zeros((synced_data.shape[0],4))
    
    print('all going as planned') 
    
    sp_mem = 0
    for i in range(0,x.shape[0]):
        if x[i,0] <= 0:
            sp_mem += 1
    sp_mem -= 1    
    
    x = x[sp_mem:x.shape[0],:]

    for j in range(1,5):        
        for i in range(0,synced_data.shape[0]):
            #La verdad es que esto es una soberana cagada porque tarda muchisimo 
            #en ejecutar el programa ya que son tres bucles for concatenados 
            #pero no se me ha ocurrido una solución mejor
            #IDEA NUEVA : DELIMITAR EL RANGO DE BUSQUEDA DEL ULTIMO A LAS 20 VALORES MAS CERCANOS AL ACTUAL
            if i <= 0:
                x_t[0,j - 1] = x[i,j]
            
            else:
                k = i + 5
                while x[k,0] <= synced_data[i, 0]:
                    k += 1
                if synced_data[i,0] == x[k, 0]:
                    x_t[i,j - 1] = x[k,j]
                else:
                    x_t[i,j - 1] = x[k - 1,j] + (synced_data[i,0]-x[k - 1,0])*(((x[k + 1,j])-x[k - 1,j])/(x[k + 1,0] - x[k - 1,0]))                     
                    
#                for k in range(i, x.shape[0]):
#                    if synced_data[i,0] == x[k, 0]:
#                        x_t[i,j - 1] = x[k,j]
#                    elif synced_data[i,0] > x[k - 1,0] and synced_data[i,0] < x[k,0]:
#                        x_t[i,j - 1] = x[k - 1,j] + (synced_data[i,0]-x[k - 1,0])*(((x[k + 1,j])-x[k - 1,j])/(x[k + 1,0] - x[k - 1,0]))
    
    for i in range(0,synced_data.shape[0]):
#        q_b_norm = math.sqrt(square(x_t[i,0]) + square(x_t[i,1]) + square(x_t[i,2]) + square(x_t[i,3]))
#        x_t[i,0] /= q_b_norm
#        x_t[i,1] /= q_b_norm
#        x_t[i,2] /= q_b_norm
#        x_t[i,3] /= q_b_norm
#        q_b = R.from_quat([x_t[i,0],x_t[i,1],x_t[i,2],x_t[i,3]])
#        eulerasos = q_b.as_euler('xyz', degrees = True)
#        synced_data[i,8] = eulerasos[0]
#        synced_data[i,9] = eulerasos[1]
#        synced_data[i,10] = eulerasos[2]
        synced_data[i,9] = x_t[i,0]
        synced_data[i,10] = x_t[i,1]
        synced_data[i,11] = x_t[i,2]
        synced_data[i,12] = x_t[i,3]
    
    for j in range(13, 16):
        for i in range(0,synced_data.shape[0]):
            synced_data[i,j] = x[i,j - 4]
    
    print('finally!')
                    
    return synced_data

#data1 = data_sync('110517','110519') 
data2 = data_sync('110901','110902')
data3 = data_sync('111247','111249') 
data4 = data_sync('111902','111902')    

A = 'Time Stamp' + ',' + 'A1' + ',' + 'A2' + ',' + 'A3' + ',' + 'A4' + ',' + 'IW' + ',' + 'IX' + ',' + 'IY' + ',' + 'IZ' + ',' + 'OW' + ',' + 'OX' + ',' + 'OY' + ',' + 'OZ' + ',' + 'XH' + ',' + 'YH' + ',' + 'ZH'  + '\n'

tdata = np.append((data2), (data3), axis = 0)
tdata = np.append((tdata), (data4), axis = 0)
print(tdata.shape)
np.savetxt(os.path.join(data_path, final_n),tdata , delimiter = ',', newline = '\n', header = A)