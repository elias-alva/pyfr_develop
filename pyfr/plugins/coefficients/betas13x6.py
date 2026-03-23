import numpy as np
import os

def pitch(r):
    file = os.path.join(os.path.dirname(__file__), '13x6-PERF.txt')

    data_lines = []
    with open(file,'r') as f:
        for line in f:
            if line.strip() == '' or not line.strip()[0].isdigit():
                continue
            if line.strip().startswith('RADIUS'):
                break
            data_lines.append(line)

    from io import StringIO
    data = np.loadtxt(StringIO(''.join(data_lines)))

    # print(data.shape)
    # print(data[:5])
    # print(data[:,0])
    # print(data[:,3])
    local_pitch = np.atan([data[:,2]/(2*np.pi*data[:,0])]) 
    # print(local_pitch*180/np.pi)
    # print((data[:,0]*2.54/100))
    xp = np.asarray(data[:,0]*2.54/100, dtype=float).ravel()
    fp1 = np.asarray(local_pitch, dtype=float).ravel()
    fp2 = np.asarray(data[:,1]*2.54/100, dtype=float).ravel()
    rinf = np.array(r[:-1])
    rsup = np.array(r[1:])
    dr = rsup-rinf
    dr = np.insert(dr,0,dr[0])
    pitch = np.interp(r, xp,fp1)
    lchord = np.interp(r, xp,fp2)

    return pitch,lchord, dr
