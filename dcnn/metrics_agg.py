
import os 
import matplotlib.pyplot as plt

def create_bar(dcnns, fusions,  col):
    cat = []
    val = [] 
    for model, metrics in dcnns.items():
        m = metrics[col] 
        cat.append(model)
        val.append(m)
    
    for model, metrics in fusions.items():
        m = metrics[col] 
        cat.append(model)
        val.append(m)

    plt.bar(cat, val)
    
    plt.xlabel('Models')
    plt.ylabel(f"{col}")
    plt.title(f"{col} Bar graph")
    plt.savefig(f"{col}_bar_graph")
    plt.clf()

path= "./metrics"
files = os.listdir(path)
metrics = {}

for file_name in files:
    if file_name.endswith(".txt") and file_name not in ["resnet50_24-10-03T00-07_metrics.txt"]:
        
        with open(f"{path}/{file_name}","r") as f: 
            met ={"precision": 0,
                  "recall"   : 0,
                  "f1-score" : 0,
                  "accuracy" : 0}
            f.readline() 
            f.readline()
            for _ in range(11):
                line = f.readline()
                m = line.split()[1:4]
                met["precision"]+= float(m[0])/11                
                met["recall"]   += float(m[1])/11              
                met["f1-score"] += float(m[2])/11
            


            f.readline()     
            
            met["accuracy"] = float(f.readline().split()[1])
            
            metrics[file_name]= met



keys = metrics.keys()
fusions = {}
base_models = {}

for k in keys:
    if "fusion" in k:
        fusions[k.split("_")[0]]=metrics[k]
    else:
        base_models[k.split("_")[0]] = metrics[k]


create_bar(base_models, fusions,  "accuracy")
print(fusions)
print()
print(base_models)




