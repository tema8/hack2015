import pdb
import sys, os
# import facerec modules
from facerec.feature import Fisherfaces, SpatialHistogram, Identity, PCA, LDA
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from dropcam import Dropcam
import urllib, json


import cv2
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

from dropcam import Dropcam


#Takes picture, stores it and displays with the message
def take_pic(cam):
    while True:
        try:
            cam.save_image("camera.0.jpg")
            #frame = (misc.imread("camera.0.jpg"))
            break
        except HTTPError: 
            print "HTTPError: Trying again."
        except:
            print "Unexpected error, trying again."





def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y,names]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            names: The corresponding name labels in a Python list
    """
    c = 0
    X,y = [], []
    names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            names.append(subject_path.split('/')[-1])

            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,names]

if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y,names] = read_images(sys.argv[1])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    #feature = Fisherfaces()
    feature = PCA()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Define the model as the combination
    #my_model = PredictableModel(feature=feature, classifier=classifier)
    ## Compute the Fisherfaces on the given data (in X) and labels (in y):
    #my_model.compute(X, y)
    ## We then save the model, which uses Pythons pickle module:
    #save_model('model.pkl', my_model)
    model = load_model('model.pkl')
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    print len(y)
    print "num eigenvectors: " + str(model.feature.eigenvectors.shape[1])
    #exit()
    for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
        E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")
    # Perform a 10-fold cross validation
    #cv = KFoldCrossValidation(model, k=10)
    #cv.validate(X, y)
    ## And print the result:
    #cv.print_results()


    #Connect to dropcam through 'cam' object
    d = Dropcam("tema8@yahoo.com", "hack2015")
    for i, cam in enumerate(d.cameras()):
        print i, " ", cam
    
    #JSON connection to the presure sensor    
    url = "https://api.particle.io/v1/devices/51ff6f065067545719150387/press?access_token=c5b16cf65dd2053b54723a33c59f3521655bf3f8"
    response = urllib.urlopen(url);
    data = json.loads(response.read())




    # Loop forever, waiting for pressure sensor signal
        # Capture image from Dropcam
        # Use OpenCV for face-detection/cropping
        # Open cropped face image and input to face recognition model
        # Do action based on identified person

    face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    print "Enter While loop"
    print names
    print y
    print "Ready to recognize"
    while True:
        #response = urllib.urlopen(url);
        #data = json.loads(response.read())

        #if data["result"] != 0:
        if raw_input("Take Picture? [y/n]: ") == 'y':
            ## Take Picture from Dropcam
            take_pic(cam)

            ## Face Detection and Cropping
            image = cv2.imread("camera.0.jpg")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) < 1:
                print "Did not have a detected face"
                continue
            face = faces[0]
            x,y,w,h = face
            face_image = image[y: y + h, x: x + w]

            # Resize to standard size
            face_image = cv2.resize(face_image, (150,150))
            cv2.imwrite("face.jpg", face_image)

            ## Display face (debugging)
            #frame = (misc.imread("face.jpg"))

            #plt.figure()
            #plt.imshow(frame)
            #plt.show()



            ## Open Face Image and Recognize
            im = Image.open("face.jpg")
            im = im.convert("L")
            Xtest = np.asarray(im, dtype=np.uint8)

            [predicted_label, classifier_output] = model.predict(Xtest)
            print "Predicted as: " + names[predicted_label]
            print "Classifier output: " + str(classifier_output) 
    
            os.system("./speech.sh Hello there, " + names[predicted_label])

            if names[predicted_label] == "Artem":
                os.system("google-chrome http://www.pandora.com &")
                os.system("./speech.sh Starting up your pandora radio, sir")
                os.system("xdotool search --onlyvisible --class 'Chrome' windowactivate")
            elif names[predicted_label] == "Jason":
                os.system("google-chrome https://www.youtube.com/embed/7IaYJZ2Usdk?autoplay=1 &")
                os.system("./speech.sh Starting up your 360-degree video, sir")

                os.system("./speech.sh You will be able to move the video by leaning left or right")
                os.system("xdotool search --onlyvisible --class 'Chrome' windowactivate")

                while True:  ## Not yet sure how to exit out of this loop
                    direction_url = "https://api.particle.io/v1/devices/53ff6d066667574848382467/lean?access_token=c5b16cf65dd2053b54723a33c59f3521655bf3f8"
                    direction_response = urllib.urlopen(direction_url);
                    direction_data = json.loads(direction_response.read())

                    if direction_data["result"] > 10:  ## Left
                        for qqq in range(20):
                            os.system("xdotool search --onlyvisible --class 'Chrome' windowfocus key 'a'")
                    elif direction_data["result"] < -10:  ## Right
                        for qqq in range(20):
                            os.system("xdotool search --onlyvisible --class 'Chrome' windowfocus key 'd'")



            elif names[predicted_label] == "Zach":
                pass

