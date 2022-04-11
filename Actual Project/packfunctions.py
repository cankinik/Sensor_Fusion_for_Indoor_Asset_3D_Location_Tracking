#-*- coding: utf-8 -*-
#Verification program for EECS 452 anchor-target-server transmission
#using manually time-stamped and encoded(TBD) UDP packets

import socket
import struct
import sys
import time

SF452_PORT = 50000 # 1000 for Windows, 50000 for Linux.
NODE_DATA_SIZE = 37
NODE_DATA_IDX = [0,NODE_DATA_SIZE,NODE_DATA_SIZE*2,NODE_DATA_SIZE*3,NODE_DATA_SIZE*4,NODE_DATA_SIZE*5]
FRAME_DATA_SIZE = 5*NODE_DATA_IDX[1]

#TST_SERVER_OUT  =   ('192.168.137.1',SF452_PORT)      #run ipconfig (Windows) or ifconfig (Linux) and replace with that address for each machine
TEST_SERVER_IN   =   ('192.168.137.1',SF452_PORT)
ANCHOR_1_IN     =   ('192.168.137.85',SF452_PORT) 
ANCHOR_2_IN     =   ('192.168.137.209',SF452_PORT)
ANCHOR_3_IN     =   ('192.168.137.28',SF452_PORT)
TARGET_1_IN     =   ('192.168.137.87',SF452_PORT)
TARGET_2_IN     =   ('192.168.137.191',SF452_PORT)
#ANCHOR_1_OUT    =   ('192.168.137.208',SF452_PORT) 
#ANCHOR_2_OUT    =   ('localhost',SF452_PORT) 
#ANCHOR_3_OUT    =   ('localhost',SF452_PORT) 

#Packs data for UDP transmission from client or server.
#Inputs:
#   data - (int, float, string) type specified by dtype
#   dtype - (string) the variable type of data
#   ptype - (string) specifies the type of information being sent.
#       Supported information types:
#           d2d - single 2-D coordinate pair
#           d3d - single 3-D coordinate pair
#           drq - request to send a single packet of data to server
#           d2p - a pair of 2-D coordinates
#           
#   tstamp[optional] - (float) epoch collection timestamp attached to the data.
def packSF452data(ptype,data=0,origin='sv',tstamp=0):
    #Decode data type into a binary stream
    if   ptype == "d2d":
        values = (ptype.encode('utf-8'),origin.encode('utf-8'),data[0],data[1],tstamp)
        packer = struct.Struct('3s 2s f f f')
        packed_data = packer.pack(*values)
    elif ptype == "d3d":
        values = (ptype.encode('utf-8'),origin.encode('utf-8'),data[0],data[1],data[2],tstamp)
        packer = struct.Struct('3s 2s f f f f')
        packed_data = packer.pack(*values)
    elif ptype == "d3p":
        values = (ptype.encode('utf-8'),origin.encode('utf-8'),data[0],data[1],data[2],data[3],data[4],data[5],tstamp)
        packer = struct.Struct('3s 2s f f f f f f f')
        packed_data = packer.pack(*values)
    elif ptype == "drq":
        values = (ptype.encode('utf-8'),origin.encode('utf-8'),data)
        packer = struct.Struct('3s 2s I')
        packed_data = packer.pack(*values)
    else:
        print("invalid packet type. Valid types are d2d, d3d, drq, d2p\n")
        sys.exit(1)
    
    return packed_data
    
    
#Sends data in a UDP datagram to address
def sendSF452data(client_socket,packet,addr):
    client_socket.sendto(packet,addr)
    return

#Pulls the latest datagram from the host_socket buffer
def recvSF452data(host_socket):
    try:
        data, address = host_socket.recvfrom(256)
    except socket.timeout:
        #print("no data received this cycle")
        return 0
    return data

def unpackSF452data(packed_data):
    #Pass through a zero if the data was not received
    if packed_data == 0:
        return 0
    unpacker_ID = struct.Struct('3s')
    data_ID_serial = packed_data[:3]
    data_ID_encoded = unpacker_ID.unpack(data_ID_serial)
    data_ID = data_ID_encoded[0].decode('utf-8')
    
    if   data_ID == "d2d":
        unpacker_all = struct.Struct("3s 2s f f f")
        data = unpacker_all.unpack(packed_data)
        return (data[0].decode('utf-8'),data[1].decode('utf-8'),data[2],data[3],data[4])
    elif data_ID == "d3d":
        unpacker_all = struct.Struct("3s 2s f f f f")
        data = unpacker_all.unpack(packed_data)
        return (data[0].decode('utf-8'),data[1].decode('utf-8'),data[2],data[3],data[4],data[5])
    elif data_ID == "drq":
        unpacker_all = struct.Struct("3s 2s I")
        data = unpacker_all.unpack(packed_data)
        return (data[0].decode('utf-8'),data[1].decode('utf-8')) 
    elif data_ID == "d3p":
        unpacker_all = struct.Struct("3s 2s f f f f f f f")
        data = unpacker_all.unpack(packed_data)
        return (data[0].decode('utf-8'),data[1].decode('utf-8'),data[2],data[3],data[4],data[5],data[6],data[7],data[8])                                               
    else:
        print("received data packet had invalid ID type\n")
        sys.exit(1)
