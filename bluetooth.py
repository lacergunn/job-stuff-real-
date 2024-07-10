import serial
import tkinter as tk
import folium as fl
import numpy as np
import io
import selenium
from PIL import Image
import cv2
from folium.plugins import MousePosition

ser = serial.Serial('COM6')  # open serial port
print(ser.name)         # check which port was really used
ser.write(b'hello')     # write a string

targetLoc=(38.803490,-76.871508)
view=2
vertRange=view*4
horzRange=view*8
m2c=1/69
boundSW=(targetLoc[0]-np.sqrt(8)/69,targetLoc[1]-np.sqrt(8)/69)
boundNE=(targetLoc[0]+np.sqrt(8)/69,targetLoc[1]+np.sqrt(8)/69)

m = fl.Map(location=targetLoc)
m.fit_bounds([boundSW,boundNE])
MousePosition().add_to(m)

img_data = m._to_png(1)
img = Image.open(io.BytesIO(img_data))
img.save('image.png')
img=cv2.imread('image.png')
if img is None:
    print("Check File Path")
print(img.shape)
centerPix = (img.shape[0]/2,img.shape[1]/2)
centerCoord=targetLoc
print(centerPix)
vp2c=vertRange/img.shape[0]/69
hp2c=horzRange/img.shape[1]/69
print(vp2c,hp2c)

# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 
    xdif=x-centerPix[1]
    ydif=y-centerPix[1]
    xCoord=centerCoord[1]-(xdif*hp2c)
    yCoord=centerCoord[0]-(ydif*vp2c)

	# checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
        print(xCoord, ' ', yCoord) 

		# displaying the coordinates 
		# on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.imshow('image', img) 

	# checking for right mouse clicks	 
    if event==cv2.EVENT_RBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
        print(xCoord, ' ', yCoord) 

		# displaying the coordinates 
		# on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[yCoord, xCoord, 0] 
        g = img[yCoord, xCoord, 1] 
        r = img[yCoord, xCoord, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (xCoord,yCoord), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 
    coordinates=str(xCoord)+','+str(yCoord)
    ser.write(bytes(coordinates, 'utf-8'))
    
if __name__=="__main__": 
  
    # reading the image 
    img = cv2.imread('image.png', 1) 
  
    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 

    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 

ser.close()             # close port

