import socket
import numpy as np
import select
import json
import serial
import time
#
#

def serial_data(port, baudrate):
    ser = serial.Serial(port, baudrate)

    while True:
        yield ser.readline()

    ser.close()


# Read IP, Port and a default ship's latitude and longitude from a file
options = "/home/gss/repos/buoy_vision/settings.py"
# Code to read RADAR TTM and GPS GPRMC messages from sockets.  GPS position and time are extracted along from GPRMC message.  RADAR range and bearing
# reading the data from the file
with open(options) as jsonFile:
    jsonData = json.load(jsonFile)

pubRate = 0.05


#---------------- End of socket setup --------------------------
# Array used to keep track of when last targetID was sent in time
target_timestamp = [0]*1001
# Boolean used to determine if BBM message should be sent to COM port

readLoop = True
pitches = np.array([0,0,0],dtype=float)
rolls = np.array([0,0,0],dtype=float)
while readLoop:
    print("Outside readloop. Entering again.")
    for line in serial_data( jsonData['serialPort'], int(jsonData['baudRate']) ):
        try:
            for dtype in (rolls,pitches):
                dtype[:-1] = dtype[1:] 
            rolls[-1] = float(repr(line).split("*")[0].split(",")[-1])
            pitches[-1] = float(repr(line).split("GPatt")[1].split(",")[2])
        except Exception as e:
            print(e)
            pass

        print("pitches:",np.average(pitches),"rolls:",np.average(rolls))
        time.sleep( pubRate )



