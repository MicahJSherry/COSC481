import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def create_bar(dcnns, fusions, col, path="metrics",
               dcnn_color="#eba834", fusion_color="#55d3fa"):
    cat = []
    val = []
    c = []
    for model, metrics in dcnns.items():
        m = metrics[col]
        cat.append(model)
        val.append(m)
        c.append(dcnn_color)
    n_dcnns = len(val)
    max_val = max(val)
    avg_val = sum(val) / n_dcnns
    c.append(dcnn_color)
    cat.append(" ")
    val.append(0)

    for model, metrics in fusions.items():
        m = metrics[col]
        cat.append(model)
        val.append(m)
        c.append(fusion_color)
    plt.bar(cat, val, color=c)
    plt.axhline(y=max_val, xmin=(n_dcnns + .5) / len(val),
                 color="b", linestyle="--", label=f"dcnn maximum {col}: {max_val:.2f}")

    plt.axhline(y=avg_val, xmin=(n_dcnns + .5) / len(val),
                 color="r", linestyle="--", label=f"dcnn average {col}: {avg_val:.2f}")
    plt.legend(loc=3)

    plt.axvline(x=n_dcnns, color="k")
    plt.xticks(rotation=23)
    plt.xlabel('Models')
    plt.ylabel(f"{col}")
    plt.title(f"{col} Bar graph")
    plt.savefig(f"{path}/svm_{col}_bar_graph")
    plt.clf()


def save_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    # Save the image
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.clf()

if __name__=="__main__":
	path = "./metrics"
	files = os.listdir(path)
	metrics = {}


	for file_name in files:
		if not(file_name.endswith(".txt") and ("svm" in file_name or "fusion" not in file_name)):
			continue	
		print(file_name)

		with open(f"{path}/{file_name}", "r") as f:
			met = {"precision": 0,
			  "recall": 0,
			  "f1-score": 0,
			  "accuracy": 0}
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
	create_bar(base_models, fusions,  "f1-score")
	create_bar(base_models, fusions,  "precision")
	create_bar(base_models, fusions,  "recall")

