import pandas as pd
import os

def readFile(filename):
    with open(filename) as file:
        lis = [line.split() for line in file] 
        return lis

filename = input('insert file name without extension: ')
_name = filename[9:len(filename)] + '.csv'
EMG_name = 'EMG' + _name
quatern_name = 'quatern' + _name
datapath = r'C:\Users\jaime\Documents\Python\TFM\Data'
filename = filename + '.csv'
nrows = pd.read_csv(os.path.join(datapath, filename)).shape[0] + 1
data = readFile(os.path.join(datapath, filename))

for i in range(0,nrows):
    data[i].pop(17)
    data[i].pop(16)
    data[i].pop(15)
    data[i].pop(14)
    data[i].pop(13)
    data[i].pop(12)
    data[i].pop(6)
    data[i].pop(0)

#print(data[10000])
A = 'Time Stamp' + ',' + 'A1' + ',' + 'A2' + ',' + 'A3' + ',' + 'A4' + ',' + 'Movement' + '\n'

for i in range(0,nrows):
        t = data[i].pop(5)
        t = t.replace(',','.')
        x = data[i].pop(0)
        x = x[1:len(x)-2]
        y = data[i].pop(0)
        y = y[1:len(y)-2]
        z = data[i].pop(0)
        z = z[1:len(z)-2]
        w = data[i].pop(0)
        w = w[1:len(w)-2]
        v = data[i].pop(0)
        v = v[1:len(v)]    
        A = A + t + ',' + y + ',' + z + ',' + w + ',' + v + '\n'
    
with open(os.path.join(datapath, EMG_name), 'w') as f:
    for line in A:
        f.write(line)
        
B = 'qw' + ',' + 'qx' + ',' + 'qy' + ',' + 'qz' + '\n'
for i in range(0,nrows):    
    xx = data[i].pop(0)
    xx = xx.replace(',','.')
    yy = data[i].pop(0)
    yy = yy.replace(',','.')
    zz = data[i].pop(0)
    zz = zz.replace(',','.')
    ww = data[i].pop(0)
    ww = ww.replace(',','.')
    mm = 'none'
    try:
        mm = data[i].pop()
    except ValueError:
        "Do nothing"    
    B = B + xx + ',' + yy + ',' + zz + ',' + ww + ',' + mm +'\n'
    
with open(os.path.join(datapath, quatern_name), 'w') as f:
    for line in B:
        f.write(line)
    