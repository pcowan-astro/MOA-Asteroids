#Architecture for MOA15 as described in Towards Asteroid Detection in Microlensing Surveys with Deep Learning by Cowan et al.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization

class MOA15(tf.keras.Model):
    def __init__(self, inputs):
        super(MOA15, self).__init__()
        
        # Block 1
        self.conv1 = Conv2D(32, (3, 3), padding='same', activation="relu")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(32, (3, 3), padding='same', activation="relu")
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(32, (3, 3), padding='same', activation="relu")
        self.bn3 = BatchNormalization()
        self.max1  = MaxPooling2D(pool_size=(2, 2), padding='same')

        # Block 2
        self.conv4 = Conv2D(64, (3, 3), padding='same', activation="relu")
        self.bn4 = BatchNormalization()
        self.conv5 = Conv2D(64, (3, 3), padding='same', activation="relu")
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(64, (3, 3), padding='same', activation="relu")
        self.bn6 = BatchNormalization()
        self.max2  = MaxPooling2D(pool_size=(2, 2), padding='same')
        
        # Block 3
        self.conv7 = Conv2D(128, (3, 3), padding='same', activation="relu")
        self.bn7 = BatchNormalization()
        self.conv8 = Conv2D(128, (3, 3), padding='same', activation="relu")
        self.bn8 = BatchNormalization()
        self.conv9 = Conv2D(128, (3, 3), padding='same', activation="relu")
        self.bn9 = BatchNormalization()
        self.max3  = MaxPooling2D(pool_size=(2, 2), padding='same')

        # Block 4
        self.conv10 = Conv2D(256, (3, 3), padding='same', activation="relu")
        self.bn10 = BatchNormalization()
        self.conv11 = Conv2D(256, (3, 3), padding='same', activation="relu")
        self.bn11 = BatchNormalization()
        self.conv12 = Conv2D(256, (3, 3), padding='same', activation="relu")
        self.bn12 = BatchNormalization()
        self.max4  = MaxPooling2D(pool_size=(2, 2), padding='same')
        
        # Block 5
        self.conv13 = Conv2D(512, (3, 3), padding='same', activation="relu")
        self.bn13 = BatchNormalization()
        self.conv14 = Conv2D(512, (3, 3), padding='same', activation="relu")
        self.bn14 = BatchNormalization()
        self.conv15 = Conv2D(512, (3, 3), padding='same', activation="relu")
        self.bn15 = BatchNormalization()
        self.max5  = MaxPooling2D(pool_size=(2, 2), padding='same')
        
        # Flatten + fully connected layer with output
        self.flat = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.bn16 = BatchNormalization()
        self.dropout1 = Dropout(0.4)
        self.fc2 = Dense(256, activation='relu')
        self.bn17 = BatchNormalization()
        self.dropout2 = Dropout(0.25)
        self.fc3 = Dense(1, activation='sigmoid')


    def call(self, inputs):
        # Block 1 
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.max1(x)

        # Block 2 
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.max2(x)

        # Block 3 
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.max3(x)
        
        # Block 4 
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.max4(x)
        
        # Block 5
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.max5(x)
        
        # fully connected and droput
        x = self.flat(x)
        x = self.fc1(x)
        x = self.bn16(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn17(x)
        x = self.dropout2(x)
        
        # output sigmoid
        x = self.fc3(x)
        
        return x
    
    def model(self):
        x = Input(shape=(128, 128, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    