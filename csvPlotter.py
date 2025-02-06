# A script to plot the CSV data whilst the model fitting and predicting is still running, using the backup files.
import csv
import matplotlib.pyplot as plt
import numpy as np

def readCsv(filename):
    with open(filename, "r+") as file:
        csv_reader = csv.reader(file)
        print(csv_reader)
        rows = list(csv_reader)[0]
        return np.asarray(rows,dtype=float)


inertias = readCsv("./MLT/cwk/inertias.txt")
print("inertias", inertias)
silhouettes = readCsv("./MLT/cwk/silhouettes.txt")
print("silhouttes", silhouettes)
kValues = readCsv("./MLT/cwk/kValue.txt")
print("KValues", kValues)
calinskiValues =  readCsv("./MLT/cwk/calinski.txt")
print("Calinski", calinskiValues)
daviesValues = readCsv("./MLT/cwk/davies.txt")
print("Davies values", daviesValues)
plt.subplot(2, 2, 1)
plt.plot(kValues, silhouettes[0:len(kValues)])
plt.title("Silhouette score")
plt.xlabel("K value")
plt.ylabel("Silhouette score")
plt.subplot(2, 2, 2)
plt.xlabel("K value")
plt.ylabel("Inertia")
# plt.scatter(validation_set.iloc[:, 0], validation_set.iloc[:, 1], c=new_labels) # selects column 0 (all rows) as x coords and column 1 (all rows) as y. and then cluster labels.
plt.plot(kValues, inertias[0:len(kValues)])
plt.title("Elbow method")
plt.subplot(2,2,3)
plt.title("Davies Bouldin score")
plt.xlabel("K value")
plt.ylabel("Davies Bouldin score")
plt.plot(kValues, daviesValues)
plt.subplot(2,2,4)
plt.title("Calinski Harabasz score")
plt.xlabel("K value")
plt.ylabel("Calinski Harabasz score")
plt.plot(kValues, calinskiValues)
# plt.savefig("./MLT/cwk/evaluation.png", dpi=fig.dpi)
plt.show()

