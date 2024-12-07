
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
#from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19

import tensorflow.keras.applications as a
from tensorflow.keras.models import Sequential

class model_factory:
    def __init__(self, num_classes=19,  in_shape=(224, 224, 3)):
        
        self.num_classes = num_classes
        self.args = {"include_top" : False,
                     "weights"     : 'imagenet',
                     "input_shape" : in_shape}
        
        self.base_models = {
            "ConvNeXtBase" : a.ConvNeXtBase,
            "ConvNeXtLarge" : a.ConvNeXtLarge,
            "ConvNeXtSmall" : a.ConvNeXtSmall,
            "ConvNeXtTiny" : a.ConvNeXtTiny,
            "ConvNeXtXLarge" : a.ConvNeXtXLarge,
            "DenseNet121" : a.DenseNet121,
            "DenseNet169" : a.DenseNet169,
            "DenseNet201" : a.DenseNet201,
            "EfficientNetB0" : a.EfficientNetB0,
            "EfficientNetB1" : a.EfficientNetB1,
            "EfficientNetB2" : a.EfficientNetB2,
            "EfficientNetB3" : a.EfficientNetB3,
            "EfficientNetB4" : a.EfficientNetB4,
            "EfficientNetB5" : a.EfficientNetB5,
            "EfficientNetB6" : a.EfficientNetB6,
            "EfficientNetB7" : a.EfficientNetB7,
            "EfficientNetV2B0" : a.EfficientNetV2B0,
            "EfficientNetV2B1" : a.EfficientNetV2B1,
            "EfficientNetV2B2" : a.EfficientNetV2B2,
            "EfficientNetV2B3" : a.EfficientNetV2B3,
            "EfficientNetV2L" : a.EfficientNetV2L,
            "EfficientNetV2M" : a.EfficientNetV2M,
            "EfficientNetV2S" : a.EfficientNetV2S,
            "InceptionResNetV2" : a.InceptionResNetV2,
            "InceptionV3" : a.InceptionV3,
            "MobileNet" : a.MobileNet,
            "MobileNetV2" : a.MobileNetV2,
            "MobileNetV3Large" : a.MobileNetV3Large,
            "MobileNetV3Small" : a.MobileNetV3Small,
            "NASNetLarge" : a.NASNetLarge,
            "NASNetMobile" : a.NASNetMobile,
            "ResNet101" : a.ResNet101,
            "ResNet101V2" : a.ResNet101V2,
            "ResNet152" : a.ResNet152,
            "ResNet152V2" : a.ResNet152V2,
            "ResNet50" : a.ResNet50,
            "ResNet50V2" : a.ResNet50V2,
            "VGG16" : a.VGG16,
            "VGG19" : a.VGG19,
            "Xception" : a.Xception
            }

            
    
    def list_models(self):
        return self.base_models.keys()
    

    def get_model(self, model_name,  train_base=True):
            
        base = self.base_models.get(model_name)(self.args)
        
        if base is None:
            raise ValueError(f"model_name: {model_name} is not defined (model_name must be one of ({base_models.keys()})") 
     
        base.trainable = train_base


        model = Sequential([
                base,
                Flatten(),
                Dense(4096,activation="relu"),
                Dropout(0.5),
                Dense(4096,activation="relu"),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ],
            name="model")


        return model


