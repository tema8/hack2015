import matplotlib.pyplot as plt
from scipy import signal, misc
from dropcam import Dropcam
import urllib, json

#Connect to dropcam through 'cam' object
d = Dropcam("tema8@yahoo.com", "hack2015")
for i, cam in enumerate(d.cameras()):
    print i, " ", cam
    

#Takes picture, stores it and displays with the message
def take_pic(cam, msg):
    cam.save_image("camera.0.jpg")
    frame = (misc.imread("camera.0.jpg"))


    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
    ax.imshow(frame)
    ax.text(0.50, 0.50, str(msg),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes,color='red', fontsize=25)

    plt.show()



#JSON connection to the presure sensor    
url = "https://api.particle.io/v1/devices/51ff6f065067545719150387/press?access_token=c5b16cf65dd2053b54723a33c59f3521655bf3f8"
response = urllib.urlopen(url);
data = json.loads(response.read())



i = 0
TIMEOUT = 1000

while i > TIMEOUT and data["result"]==0:
    i = i+1
    response = urllib.urlopen(url);
    data = json.loads(response.read())

if i >= TIMEOUT:
    take_pic(cam, "Time Out")
else:    
    take_pic(cam, "Hey, you!")