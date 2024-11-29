
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
#from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19

import tensorflow.keras.applications as a
from tensorflow.keras.models import Sequential

class model_factory:
    def __init__(self, in_shape=(224, 224, 3)):
         
        self.base_models = {
            #resnet V1
            "resnet50" : a.ResNet50(include_top=False, weights='imagenet', input_shape=in_shape),
            "resnet101" : a.ResNet101(include_top=False, weights='imagenet', input_shape=in_shape),
            "resnet152" : a.ResNet152(include_top=False, weights='imagenet', input_shape=in_shape),
            
            #resnet V2
            "resnet50v2"  : a.ResNet50V2(include_top=False, weights='imagenet', input_shape=in_shape),
            "resnet101v2" : a.ResNet101V2(include_top=False, weights='imagenet', input_shape=in_shape),
            "resnet152v2" : a.ResNet152V2(include_top=False, weights='imagenet', input_shape=in_shape),
            
            #EffecientNet V1
            "EfficientNetB0" : a.EfficientNetB0(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB1" : a.EfficientNetB1(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB2" : a.EfficientNetB2(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB3" : a.EfficientNetB3(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB4" : a.EfficientNetB4(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB5" : a.EfficientNetB5(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB6" : a.EfficientNetB6(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB7" : a.EfficientNetB7(include_top=False, weights='imagenet', input_shape=in_shape),

            #EffecientNet V2
            "EfficientNetB0v2" : a.EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB1v2" : a.EfficientNetV2B1(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB2v2" : a.EfficientNetV2B2(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetB3v2" : a.EfficientNetV2B3(include_top=False, weights='imagenet', input_shape=in_shape),
             
            "EfficientNetv2L" : a.EfficientNetV2L(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetv2M" : a.EfficientNetV2M(include_top=False, weights='imagenet', input_shape=in_shape),
            "EfficientNetv2S" : a.EfficientNetV2L(include_top=False, weights='imagenet', input_shape=in_shape),
             
            #other 
            "googlenet" : a.InceptionV3(include_top=False, input_shape=in_shape),
            "vgg19" : a.VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=in_shape),
            "vgg16" : a.VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=in_shape),
            }
    
    def list_models(self):
        return self.base_models.keys()
    

    def get_model(self, model_name="resnet50", num_classes=11, train_base=True):
        
            
        base = self.base_models.get(model_name)
        
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
                Dense(num_classes, activation='softmax')
            ],
            name="model")


        return model


