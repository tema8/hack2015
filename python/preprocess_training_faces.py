import sys, os
import numpy as np
import cv2


def crop_faces(path):
    face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    for dirname, dirnames, filenames in os.walk(path):    
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    image = cv2.imread(os.path.join(subject_path, filename))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) < 1:
                        print os.path.join(subject_path, filename) + " did not have a detected face"
                        os.remove(os.path.join(subject_path, filename))
                        continue
                    face = faces[0]
                    x,y,w,h = face
                    face_image = image[y: y + h, x: x + w]

                    # Resize to standard size
                    face_image = cv2.resize(face_image, (150,150))

                    cv2.imwrite(os.path.join(subject_path, filename), face_image)

                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise




def printUsage():
    print "python " + sys.argv[0] + " /path/to/training/data/"


if __name__ == "__main__":

    if len(sys.argv) < 2:
        printUsage()
        exit()

    data_path = sys.argv[1]

    crop_faces(data_path)


