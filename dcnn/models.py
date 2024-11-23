
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19
from tensorflow.keras.models import Sequential

def model_factory(model_name="resnet50", num_classes=11):
	
	base_models = {
		"resnet50" : ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
		"googlenet": InceptionV3(include_top=False, input_shape=(224, 224, 3)),
 		"vgg19" : VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=(224, 224, 3)),
 		"vgg16" : VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=(224, 224, 3)),
		}
	base = base_models.get(model_name)
	
	if base is None:
		raise ValueError(f"model_name: {model_name} is not defined (model_name must be one of ({base_models.keys()})") 
 
	base.trainable = False


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


