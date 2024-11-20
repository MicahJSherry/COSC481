import os 
import tensorflow as tf
import numpy as np
features_path = "./features"

from load_image import load_images

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def save_features(images_dir="./spark22/train", out_dir= features_path):
    num_images = -1
    X, y = load_images("./spark22/train",num_images)
    print(y)
    
    model_paths = [
        "models/googlenet_24-11-18T16-07.keras",
        "models/resnet50_24-11-18T16-07.keras",
        "models/vgg16_24-11-18T16-07.keras",
        "models/vgg19_24-11-18T16-07.keras"]
    

    model_features = {}
    for p in model_paths:
        model = tf.keras.models.load_model(p)
        model.trainable = False
        model = tf.keras.Sequential(model.layers[:-1]) 
        features = model.predict(X)
        
        architecture_name = p.split("/")[1].split("_")[0] 
        print( "\t" + architecture_name)
        model_features[architecture_name] = features 
        
        for  i in range(len(features)):
            feature = features[i]
            path= f"{features_path}/{architecture_name}/{y[i]}"
            os.makedirs(path,exist_ok= True)
            with open(f'{path}/{i}.npy', 'wb') as f:
                np.save(f, feature)

    return model_features, y

def load_features(root_dir=features_path):
    features = {}
    y = []
    for model_dir in os.listdir(root_dir):
               
        if not os.path.isdir(f"{root_dir}/{model_dir}"):
            continue
        X = []
        y = []

        for class_dir in os.listdir(f"{root_dir}/{model_dir}"):
            print(f"{root_dir}/{model_dir}/{class_dir}")

            if not os.path.isdir(f"{root_dir}/{model_dir}/{class_dir}"):
                continue
            

            for file in os.listdir(f"{root_dir}/{model_dir}/{class_dir}"):
                if not file.endswith(".npy"):
                    continue
                x = np.load(f"{root_dir}/{model_dir}/{class_dir}/{file}")
                X.append(x)
                y.append(class_dir)

        if len(X) != 0:
            features[model_dir]= np.array(X)
    
    return features, y

def fuse(features, method="cat", dim_red="pca"):
    features_array = [features[model] for model in features.keys()]

    methods = {
        "cat" : np.hstack,
        "mult": np.multiply.reduce,
        "add": np.add.reduce,
        "min":np.minimum.reduce,
        "max":np.maximum.reduce,
        "avg":np.frompyfunc(lambda x, y,: (x+y)/2, 2, 1).reduce
            }
    
    dim_reducers ={
        "pca" : PCA(n_components=16),
        "tsne": TSNE(n_components=3)
            }
    
    meth = methods.get(method)
    reducer = dim_reducers.get(dim_red)


    if meth is None:
        raise ValueError(f"method: {method} is not defined")
    
    


    x = meth(features_array) 
    if reducer is not None:  
        x = reducer.fit_transform(x)
    print(x.shape) 
    return x 
        
if len(os.listdir(features_path)) == 0:
    print("saveing the models features.")
    features, y = save_features()
else:
    print("loading the saved features.")
    features, y = load_features()

methods = ["avg"]

reductions =["pca", "tsne"]

for m in methods:
    for r in reductions:
        x = fuse(features, method=m, dim_red=r)


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        kernal = "rbf"
        svm = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal,verbose=2))

        svm.fit(x_train, y_train)

        y_pred = svm.predict(x_test)
        metrics = classification_report(y_true=y_test,y_pred=y_pred)
        
        with open(f"metrics/{m}-{r}_fusion_svm_metric.txt", "w") as f:
            f.write(str(metrics))





