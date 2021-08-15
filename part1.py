import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow import keras
from keras.layers import Input ,Dense,Activation, Conv2D,AveragePooling2D,Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_images():

   images1 = []
   X1=[]
   Y1=[]
   Y2=[]
   Y3=[]
   for filename in os.listdir('blue1'):
      img = cv2.imread(os.path.join('blue1',filename))
      if img is not None:
         images1.append(img)
         #print(img.shape)
         X1.append(img)
         Y1.append(1)
         Y2.append(0)
         Y3.append(0)


   images2 = []
   for filename in os.listdir('red1'):
      img = cv2.imread(os.path.join('red1',filename))
      if img is not None:
         images2.append(img)
         X1.append(img)
         Y1.append(0)
         Y2.append(1)
         Y3.append(0)

   images3 = []
   for filename in os.listdir('referee'):
      img = cv2.imread(os.path.join('referee',filename))
      if img is not None:
         images3.append(img)
         X1.append(img)
         Y1.append(0)
         Y2.append(0)
         Y3.append(1)

   for i in range(len(X1)):
      X1[i] = cv2.resize(X1[i], (50,50), interpolation = cv2.INTER_AREA)

   d = {'1':Y1,'2':Y2,'3':Y3}
   y=pd.DataFrame(d, columns=['1','2','3'])
   X1=np.array(X1)
   x_train, x_test, y_train, y_test = train_test_split(X1, y)
   x_train=np.array(x_train)
   y_train=np.array(y_train)
   x_test=np.array(x_test)
   y_test=np.array(y_test)
   x_train = x_train.astype('float32')/255
   x_test = x_test.astype('float32')/255

   return x_train, x_test, y_train, y_test


def build_model(input_shape):
  
  x_input = Input(shape =input_shape,name = 'input')

  x = Conv2D(filters = 16,kernel_size = (2,2), strides = 1, padding = 'valid',name = 'conv2')(x_input)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size =2,strides = 2,name = 'pad2')(x)

  x = Flatten()(x)

  x = Dense(units = 120, name = 'fc_1')(x)

  x = Activation('relu', name = 'relu_1')(x)
  # x = Dropout(rate = 0.5)

  x = Dense(units =84, name = 'fc_2')(x)
  x = Activation('relu', name = 'relu_2')(x)
  # x = Dropout(rate = 0.5)

  outputs = Dense(units = 3,name = 'softmax', activation='softmax')(x)
  
  model = Model(inputs = x_input, outputs = outputs)
  model.summary()

  return model


def train_model(x_train,x_test,y_train,y_test):

   model = build_model(input_shape=(50,50,3))
   model.compile(optimizer = 'adam',loss ='binary_crossentropy' ,metrics = ['accuracy'])
   datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
   batch_size = 5

   
   model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
   validation_data=(x_test, y_test),                       
      steps_per_epoch=len(y_train) // batch_size, epochs=40)

   return model
   #reconstructed_model = keras.models.load_model("my_model_2")


def detection_classification(model):

   cap1 = cv2.VideoCapture('0057_2013-11-03 18-01-17.249311000_2.h264')
   cap2 = cv2.VideoCapture('0057_2013-11-03 18-01-17.249311000.h264')
   cap3 = cv2.VideoCapture('0057_2013-11-03 18-01-17.249311000_3.h264')

   I=cv2.imread('2D_field.png')
   fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
   i=0
   buffer=[]
   INPUT_SHAPE=(50,50)
   while(1):
      
      ret1, frame1 = cap1.read()
      if ret1 == False: # end of video (perhaps)
         print(1)
         break
      
      ret2, frame2 = cap2.read()
      if ret2 == False: # end of video (perhaps)
         print(1)
         break  
      ret3, frame3 = cap3.read()
      if ret3 == False: # end of video (perhaps)
         print(1)
         break 

      J1 = frame1.copy()
      points1 = np.float32([[143,164],
                        [642,107],
                        [1136,117],
                        [873,775]])
      points2=np.array([(164,139),(525,0),(886,139),(525,680)], dtype=np.float32)
      H1 = cv2.getPerspectiveTransform(points1,points2)
      output_size = (I.shape[1], I.shape[0])

      J2 = frame2.copy()
      points1 = np.float32([[827,149],
                        [899,185],
                        [478,291],
                        [55,299]])
      points2=np.array([(0,0),(164,139),(164,541),(0,680)], dtype=np.float32)
      H2 = cv2.getPerspectiveTransform(points1,points2)
      output_size = (I.shape[1], I.shape[0])

      J3 = frame3.copy()
      points1 = np.float32([[452,208],
                        [878,281],
                        [406,902],
                        [1258,263]])
      points2=np.array([(886,139),(886,541),(525,680),(1050,680)], dtype=np.float32)
      H3 = cv2.getPerspectiveTransform(points1,points2)
      output_size = (I.shape[1], I.shape[0])

      fgmask1 = fgbg.apply(frame1)
      fgmask2 = fgbg.apply(frame2)
      fgmask3 = fgbg.apply(frame3)
      threshold = 100
      ret1, T1 = cv2.threshold(fgmask1,threshold,255,cv2.THRESH_BINARY)
      ret2, T2 = cv2.threshold(fgmask2,threshold,255,cv2.THRESH_BINARY)
      ret3, T3 = cv2.threshold(fgmask3,threshold,255,cv2.THRESH_BINARY)

      kernel1=[[1,0,1,0,1],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [0,1,1,1,0],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [0,1,1,1,0],
               [1,1,1,1,1],
               [1,1,1,1,1],
               [1,0,1,0,1]]
      kernel1=np.array(kernel1,dtype=np.uint8)
      I1=I.copy()

      
      
      def my_function(TT,H,frame,nn):

         ## closing
         T = cv2.morphologyEx(TT, cv2.MORPH_CLOSE, kernel1)
         n,C,stats, centroids = cv2.connectedComponentsWithStats(T)
         positions=[]
         color=[]

         for i in range(1,n):
            width = stats[i][2]
            height = stats[i][3]
            area = stats[i][4]
            if(height>20 and  area>25 and height >= width):
               try:
                  # cv2.circle(frame1, (int(centroids[i][0]), int(centroids[i][1]+height/2)), 3, [0,0,255],10)
                  top_left = (int(centroids[i][0]-width/2), int(centroids[i][1]-height/2))
                  bot_right = (int(centroids[i][0]+width/2), int(centroids[i][1]+height/2))
                  bot_center = (int(centroids[i][0]), int(centroids[i][1]+height/2))
                  
                  #cv2.rectangle(frame, top_left, bot_right, thickness=3, color=[0,0,255])
                  if(nn==1 or (nn==2 and bot_center[0]>155 and bot_center[1]<250 and bot_center[0]>=bot_center[1]) or 
                     (nn==3 and bot_center[0]>700)):
                     a=(frame[top_left[1]:bot_right[1],top_left[0]:bot_right[0],:])
                     a= cv2.resize(a, (50,50), interpolation = cv2.INTER_AREA)
                  
                     sample=np.array(a)
                     sample = sample.astype('float32')/255

                     batch = np.expand_dims(sample, axis=0)
                     prediction = model.predict(batch)
                     
                     for i in range(0,3):
                        if prediction[0][i]<=0.5 :
                           prediction[0][i]=0
                        else:
                           prediction[0][i]=1
                     #print(prediction)
                     if(prediction[0][0]==1): 
                        color.append(1)
                        positions.append(bot_center)
                     elif (prediction[0][1]==1): 
                        color.append(2)
                        positions.append(bot_center)
                     elif (prediction[0][2]==1):
                        color.append(3) 
                        positions.append(bot_center)
                     elif (prediction[0][0]==2) :
                        positions.append(bot_center)

               except:
                  
                  break
               
         if len(positions) != 0:
            positions = np.float32(positions).reshape(-1, 1, 2)
            map_positions = cv2.perspectiveTransform(positions,H).reshape(-1, 2)
            
            for i in range(len(map_positions)):

                  if(color[i]==1):
                     cv2.circle(I1, (int(map_positions[i][0]), int(map_positions[i][1])), color=(255,0,0), radius = 4, thickness=4)
                  elif(color[i]==2):
                     cv2.circle(I1, (int(map_positions[i][0]), int(map_positions[i][1])), color=(0,0,255), radius = 4, thickness=4)
                  elif(color[i]==3):
                     cv2.circle(I1, (int(map_positions[i][0]), int(map_positions[i][1])), color=(0,255,0), radius = 4, thickness=4)

      my_function(T1,H1,frame1,1)
      my_function(T2,H2,frame2,2)
      my_function(T3,H3,frame3,3)

      #cv2.imshow('frame',frame1)
      cv2.imshow('warped',I1)
      k = cv2.waitKey(10) & 0xff
      
      i+=1
      if k == ord('q'):
            break
   cap1.release()
   cap2.release()
   cap3.release()




def main():
    x_train,x_test,y_train,y_test=read_images()
    model=train_model(x_train,x_test,y_train,y_test)
    detection_classification(model)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
