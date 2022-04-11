# -*- coding: utf-8 -*-
#Test server for EECS 452 Sensor Fusion WN2022
import socket
import struct
import cv2 as cv
from packfunctions import *
#import numpy as np
import time
import numpy as np

s_IN1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_IN2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_IN2.bind(TEST_SERVER_IN)
NODE_LIST = (ANCHOR_1_IN,ANCHOR_2_IN,ANCHOR_3_IN)#, TARGET_1_IN, TARGET_2_IN)




currdata = bytearray(FRAME_DATA_SIZE) #Size of each frame data packet in bytes
request_cmd = packSF452data('drq')     #Create a request packet
#s_IN.settimeout(5)                     #Maximum wait time for requested packet
data_accross_time = []
number_of_sources = 3
while True:
    try:
        data_frame = np.zeros((number_of_sources, 2, 3))    # source, obj, xyz
        for n in range(number_of_sources):
            start_time = time.time()
            newdata = 0     #Initialize new data as not received
            s_IN1.connect(NODE_LIST[n])
            sendSF452data(s_IN1,request_cmd,NODE_LIST[n])
            newdata = recvSF452data(s_IN2)
            if newdata != 0:
                currdata[NODE_DATA_IDX[n]:NODE_DATA_IDX[n+1]] = newdata
                cycle_time = time.time() - start_time
                # print("Packet received from client %d in %.6f seconds" % (n+1,cycle_time))
                packet1 = unpackSF452data(newdata)
                # print("Content ID: %s   Source: %s" % (packet1[0],packet1[1]))
                # print("x1=%.6f y1=%.6f z1=%.6f x2=%.6f y2=%.6f z2=%.6f" % (packet1[2],packet1[3],packet1[4],packet1[5],packet1[6],packet1[7]))
                # print("Timestamp: %.3f s\n" % packet1[8])
                data_frame[int(packet1[1][1])-1] = np.array([[packet1[2],packet1[3],packet1[4]], [packet1[5],packet1[6],packet1[7]]])
        data_accross_time.append(data_frame)
        print(data_frame)
    except KeyboardInterrupt:        
        break
final_data = np.array(data_accross_time)
with open('C:/Users/canki/OneDrive/Desktop/data_from_all_sources.npy', 'wb') as f:
    np.save(f, final_data)
print('Saved the data!')
        


            
        
    
    #Only replace the data that did arrive
    # log_ptr = open("clog.txt",'ab')
    # log_ptr.write(currdata)
    #log_ptr.close()
    #TODO: Keyboard Interrupt
    

s_IN1.close()
s_IN2.close()
# log_ptr.close()    