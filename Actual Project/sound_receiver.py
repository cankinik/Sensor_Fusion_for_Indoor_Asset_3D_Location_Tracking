from sys import byteorder
from array import array
from struct import pack
from scipy.signal import chirp, spectrogram
import numpy as np
from scipy import stats
import pyaudio
import wave
import matplotlib.pyplot as plt
import time

import socket
from packfunctions import *

s_COMMAND = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s_COMMAND.bind(TARGET_1_IN)
s_COMMAND.settimeout(None)


fs = 44100
T = 0.05
t = np.linspace(0, T,int(T*fs))
c1 = chirp(t, f0=14000, f1=16000, t1=T, method='linear')
c2 = chirp(t, f0=16000, f1=18000, t1=T, method='linear')
c3 = chirp(t, f0=18000, f1=20000, t1=T, method='linear')

THRESHOLD = 20000
CHUNK_SIZE = int(fs * T)
FORMAT = pyaudio.paInt16
RATE = 44100

def TDoA(Xb, X, RD, X_estimate, Y_estimate):
    BSN = np.shape(X)[0]
    x = X_estimate
    y = Y_estimate

    MS = [x,y]
    iEP = MS

    Q = 0.01 * np.eye(BSN)
    delta = [1,1]
    kk = 0
    while (np.abs(delta[0]) + np.abs(delta[1])) > 0.01:
        R1 = np.sqrt((iEP[0]-Xb[0])**2 + (iEP[1]-Xb[1])**2 )
        R = np.zeros(BSN)
        kk = kk+1
        for i in range(BSN):
            R[i] = np.sqrt((iEP[0]-X[i,0])**2 + (iEP[1]-X[i,1])**2 )

        hi = np.zeros(BSN)
        for i in range(BSN):
            hi[i] = RD[i] - (R[i]-R1)

        Gi = np.zeros([BSN,2])
        for i in range(BSN):
            Gi[i,0] = (Xb[0]-iEP[0])/R1 - (X[i,0]-iEP[0])/R[i]
            Gi[i,1] = (Xb[1]-iEP[1])/R1 - (X[i,1]-iEP[1])/R[i]

        delta = np.linalg.inv(np.transpose(Gi)@np.linalg.inv(Q)@Gi)@ np.transpose(Gi)@np.linalg.inv(Q)@hi
        if (np.abs(delta[0]) + np.abs(delta[1])) > 0.01:
            EP = iEP + np.transpose(delta)
            iEP = EP

    return iEP
    

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    comb = array('h')
    temp_prev2 = np.zeros(CHUNK_SIZE)
    temp_prev1 = np.zeros(CHUNK_SIZE)
    temp_curr = np.zeros(CHUNK_SIZE)
    
    cali_flag = 1
    estimate_flag = 0
    
    a1_odd_even = -1
    a2_odd_even = -1
    a3_odd_even = -1
    new_a1 = 0
    new_a2 = 0
    new_a3 = 0
    cali_21 = 0
    estimate_delta21 = 0
    estimate_delta31 = 0
    cali_31 = 0
    cali_iter_start = 30
    cali_iter_stop = 50
    cali_a1 = np.zeros([2,cali_iter_stop - cali_iter_start])
    cali_a2 = np.zeros([2,cali_iter_stop - cali_iter_start])
    cali_a3 = np.zeros([2,cali_iter_stop - cali_iter_start])
    flag_frame = 0
    number_a1 = 0
    number_a2 = 0
    number_a3 = 0
    while 1:
        flag_frame += 1

        snd_data = array('h', stream.read(CHUNK_SIZE))
        r.extend(snd_data)
        temp_prev2 = temp_prev1
        temp_prev1 = temp_curr
        temp_curr = snd_data

        if cali_flag == 1:
            comb1 = np.append(temp_prev2[int((CHUNK_SIZE-1)/2):-1],temp_prev1)
            comb = np.append(comb1,temp_curr[0:int((CHUNK_SIZE+1)/2)])
    
            corr1 = np.abs(np.correlate(comb,c1,"full"))
            if np.max(corr1) > THRESHOLD:
                if np.argmax(corr1) >= 2205 and np.argmax(corr1) <= 4410:
                    number_a1 += 1
                    print(np.argmax(corr1))
                    print('number of anchor1:',number_a1)
                    #print(np.argmax(corr1))
                    if number_a1 >= cali_iter_start and number_a1 < cali_iter_stop:
                        cali_a1[0,number_a1-cali_iter_start] = flag_frame
                        cali_a1[1,number_a1-cali_iter_start] = np.argmax(corr1)
                        print(cali_a1)
                                           
            corr2 = np.abs(np.correlate(comb,c2,"full"))
            if np.max(corr2) > THRESHOLD:
                if np.argmax(corr2) >= 2205 and np.argmax(corr2) <= 4410:
                    number_a2 += 1
                    print(np.argmax(corr2))
                    print('number of anchor2:',number_a2)
                    if number_a2 >= cali_iter_start and number_a2 < cali_iter_stop:
                        cali_a2[0,number_a2-cali_iter_start] = flag_frame
                        cali_a2[1,number_a2-cali_iter_start] = np.argmax(corr2)
                        print(cali_a2)
            corr3 = np.abs(np.correlate(comb,c3,"full"))
            if np.max(corr3) > THRESHOLD:
                if np.argmax(corr3) >= 2205 and np.argmax(corr3) <= 4410:
                    number_a3 += 1
                    print('number of anchor3:',number_a3)
                    print(np.argmax(corr3))
                    #print(np.argmax(corr3))
                    if number_a3 >= cali_iter_start and number_a3 < cali_iter_stop:
                        cali_a3[0,number_a3-cali_iter_start] = flag_frame
                        cali_a3[1,number_a3-cali_iter_start] = np.argmax(corr3)
                        #print(np.argmax(corr3))
                        #print(flag_frame)
                        print(cali_a3)
            if number_a1 >= cali_iter_stop and number_a2 >= cali_iter_stop and number_a3 >= cali_iter_stop:
                
                a1_odd_even = stats.mode(np.remainder(cali_a1[0,:],2))[0][0]
                a2_odd_even = stats.mode(np.remainder(cali_a2[0,:],2))[0][0]
                a3_odd_even = stats.mode(np.remainder(cali_a3[0,:],2))[0][0]

                for i in range(cali_iter_stop - cali_iter_start):
                    if cali_a1[0,-i+cali_iter_stop - cali_iter_start-1] % 2 != a1_odd_even:
                        cali_a1 = np.delete(cali_a1,-i+cali_iter_stop - cali_iter_start-1,axis=1)
                    if cali_a2[0,-i+cali_iter_stop - cali_iter_start-1] % 2 != a2_odd_even:
                        cali_a2 = np.delete(cali_a2,-i+cali_iter_stop - cali_iter_start-1,axis=1)
                    if cali_a3[0,-i+cali_iter_stop - cali_iter_start-1] % 2 != a3_odd_even:
                        cali_a3 = np.delete(cali_a3,-i+cali_iter_stop - cali_iter_start-1,axis=1)
                
                cali_21 = np.mean(cali_a2[1,:]) - np.mean(cali_a1[1,:])
                cali_31 = np.mean(cali_a3[1,:]) - np.mean(cali_a1[1,:])
                
                cali_flag = 0
                estimate_flag = 1
        elif estimate_flag == 1:
            if a1_odd_even == 0:
                if flag_frame%2 == 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c1,"full"))) > THRESHOLD:
                        new_a1 = np.argmax(np.abs(np.correlate(combb,c1,"full")))

            elif a1_odd_even > 0:
                if flag_frame%2 != 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c1,"full"))) > THRESHOLD:
                        new_a1 = np.argmax(np.abs(np.correlate(combb,c1,"full")))

            if a2_odd_even == 0:
                if flag_frame%2 == 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c2,"full"))) > THRESHOLD:
                        new_a2 = np.argmax(np.abs(np.correlate(combb,c2,"full")))
            elif a2_odd_even > 0:
                if flag_frame%2 != 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c2,"full"))) > THRESHOLD:
                        new_a2 = np.argmax(np.abs(np.correlate(combb,c2,"full")))

            if a3_odd_even == 0:
                if flag_frame%2 == 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c3,"full"))) > THRESHOLD:
                        new_a3 = np.argmax(np.abs(np.correlate(combb,c3,"full")))
            elif a3_odd_even > 0:
                if flag_frame%2 != 0:
                    comb11 = np.append(temp_prev2,temp_prev1)
                    combb = np.append(comb11,temp_curr)
                    if np.max(np.abs(np.correlate(combb,c3,"full"))) > THRESHOLD:
                        new_a3 = np.argmax(np.abs(np.correlate(combb,c3,"full")))

        if cali_flag == 0 and new_a1 != 0 and new_a2 != 0 and new_a3 != 0:
            estimate_delta21 = new_a2 - new_a1 - cali_21
            estimate_delta31 = new_a3 - new_a1 - cali_31

            Xb = [0,0]
            X = np.matrix([[2.81,0],[2.81,1.9]])
            X_estimate = 1
            Y_estimate = 1

            RD = [estimate_delta21/fs*343,estimate_delta31/fs*343]
            #print(RD)
            
            try:
                pos = TDoA(Xb, X, RD, X_estimate, Y_estimate)
                
            except Exception:
                pos = [-100, -100]          # outliers
                pass
            
            print(pos)
            packed_data = packSF452data('d3p',[pos[0],pos[1],-100,-100,-100,-100],'a4')
            junk = recvSF452data(s_COMMAND)
            sendSF452data(s_COMMAND,packed_data,TEST_SERVER_IN)
        
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    print("Recording starts")
    record_to_file('demo.wav')
    print("done - result written to demo.wav")
