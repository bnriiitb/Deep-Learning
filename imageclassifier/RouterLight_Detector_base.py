# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib
import os
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt 


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation/'
test_data_dir = 'data/test/'

nb_epoch = 50
nb_train_samples = 331
nb_validation_samples = 61

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255, # scale images
                            zoom_range=0.01, # randomly zoom into images 1% of images
                            width_shift_range=0.1,# randomly shift images horizontally (10% of total width)
                            height_shift_range=0.1,# randomly shift images vertically (10% of total height)
                            horizontal_flip=True)# randomly flip images horizontally


 
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=25,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=25,
        class_mode='binary')
    
# Load the Keras Model for Router Detection
print('Loading the Model from disk.')
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
print('Loading model weighta')
loaded_model.load_weights("models/basic_cnn_30_epochs.h5")
print("Successfully loaded the model from disk.")

print('Model compilation started')
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Model compilation completed successfully')

#loss,accuracy=loaded_model.evaluate_generator(validation_generator, nb_validation_samples)
#print('###### Model Evaluation Metrics ######')
#print('Loss: ',loss)
#print('Accuracy: ',accuracy)
    
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'Red':(166, 84, 141), 'Green':(33,80,40),'orange':(0, 41, 75)} 
#assign new item lower['blue'] = (93, 10, 0)
upper = {'Red':(186,255,255), 'Green':(102,255,255),'orange':(20,255,255)}

# define standard colors for circle around the object
colors = {'Red':(0,0,255), 'Green':(0,255,0),'orange':(0,140,255)}

#pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    
 
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    #frame = cv2.imread('/Users/panda/Pictures/router.png')
 
    # resize the frame, blur it, and convert it to the HSV
    # color space
    #print('Classes: ',train_generator.class_indices)
   
    #frame = imutils.resize(frame, width=150)
    test_image = image.img_to_array(frame)
    test_image = cv2.resize(test_image, (150,150))
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'other'
        cv2.putText(frame,"Please point camera towards router", (175,175), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors['Red'],2)
    else:
        prediction = 'gen3vz'
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #for each color in dictionary check object in frame
        for key, value in upper.items():
            # construct a mask for the color from dictionary`1, then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
                # only proceed if the radius meets a minimum size. Correct this value for your obect's size
                if radius > 0.001:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                    cv2.putText(frame,key + " Light", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                    if(key=='Green'):
                        print('Seems Everythng Okay')
                    elif(key=='Red'):
                        print('Seems there is some problem with your router')
    print('The object in the picture is ',prediction)

     
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 
#cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()