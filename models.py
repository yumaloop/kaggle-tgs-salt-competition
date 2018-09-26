from keras import backend as K
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import array_to_img, img_to_array, load_img

class Unet_resnet():
    def __init__(self, img_width=101, img_height=101, img_ch=1, first_filters=16, dropout_rate=0.5):
        input_layer = Input((img_width, img_height, img_ch))
        output_layer = self.make_model(input_layer, first_filters, dropout_rate)
        self.model = Model(input_layer, output_layer)

    def build_model(self): 
       return self.model    

    def BatchActivate(self, x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def convolution_block(self, x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation == True:
            x = self.BatchActivate(x)
        return x

    def residual_block(self, blockInput, num_filters=16, batch_activate = False):
        x = self.BatchActivate(blockInput)
        x = self.convolution_block(x, num_filters, (3,3) )
        x = self.convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        if batch_activate:
            x = self.BatchActivate(x)
        return x

    def make_model(self, input_layer, first_filters, dropout_rate = 0.5):

        ''' Down-Sampling '''

        # 101 -> 50
        conv1 = Conv2D(first_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
        conv1 = self.residual_block(conv1, first_filters * 1)
        conv1 = self.residual_block(conv1, first_filters * 1, True)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(dropout_rate/2)(pool1)

        # 50 -> 25
        conv2 = Conv2D(first_filters * 2, (3, 3), activation=None, padding="same")(pool1)
        conv2 = self.residual_block(conv2, first_filters * 2)
        conv2 = self.residual_block(conv2, first_filters * 2, True)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(dropout_rate)(pool2)

        # 25 -> 12
        conv3 = Conv2D(first_filters * 4, (3, 3), activation=None, padding="same")(pool2)
        conv3 = self.residual_block(conv3, first_filters * 4)
        conv3 = self.residual_block(conv3, first_filters * 4, True)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(dropout_rate)(pool3)

        # 12 -> 6
        conv4 = Conv2D(first_filters * 8, (3, 3), activation=None, padding="same")(pool3)
        conv4 = self.residual_block(conv4, first_filters * 8)
        conv4 = self.residual_block(conv4, first_filters * 8, True)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(dropout_rate)(pool4)


        ''' Middle '''

        convm = Conv2D(first_filters * 16, (3, 3), activation=None, padding="same")(pool4)
        convm = self.residual_block(convm, first_filters * 16)
        convm = self.residual_block(convm, first_filters * 16, True)


        ''' Up-Sampling '''

        # 6 -> 12
        deconv4 = Conv2DTranspose(first_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(dropout_rate)(uconv4)

        uconv4 = Conv2D(first_filters * 8, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = self.residual_block(uconv4, first_filters * 8)
        uconv4 = self.residual_block(uconv4, first_filters * 8, True)

        # 12 -> 25
        deconv3 = Conv2DTranspose(first_filters * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
        uconv3 = concatenate([deconv3, conv3])    
        uconv3 = Dropout(dropout_rate)(uconv3)

        uconv3 = Conv2D(first_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = self.residual_block(uconv3, first_filters * 4)
        uconv3 = self.residual_block(uconv3, first_filters * 4, True)

        # 25 -> 50
        deconv2 = Conv2DTranspose(first_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])

        uconv2 = Dropout(dropout_rate)(uconv2)
        uconv2 = Conv2D(first_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = self.residual_block(uconv2, first_filters * 2)
        uconv2 = self.residual_block(uconv2, first_filters * 2, True)

        # 50 -> 101
        deconv1 = Conv2DTranspose(first_filters * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
        uconv1 = concatenate([deconv1, conv1])

        uconv1 = Dropout(dropout_rate)(uconv1)
        uconv1 = Conv2D(first_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = self.residual_block(uconv1, first_filters * 1)
        uconv1 = self.residual_block(uconv1, first_filters * 1, True)
        
        output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
        output_layer =  Activation('sigmoid')(output_layer_noActi)

        return output_layer





