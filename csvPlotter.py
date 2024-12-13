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
silhouettes = readCsv("./MLT/cwk/silhouttes.txt")
print("silhouttes", silhouettes)
kValues = readCsv("./MLT/cwk/kValue.txt")
print("KValues", kValues)

fig = plt.figure(figsize=(11,6))
plt.subplot(1, 2, 1)
plt.plot(kValues, silhouettes[0:len(kValues)])
plt.title("Silhoutte scores")
plt.xlabel("K value")
plt.ylabel("Silhoutte score")
plt.subplot(1, 2, 2)
plt.xlabel("K value")
plt.ylabel("Inertia")
# plt.scatter(validation_set.iloc[:, 0], validation_set.iloc[:, 1], c=new_labels) # selects column 0 (all rows) as x coords and column 1 (all rows) as y. and then cluster labels.
plt.plot(kValues, inertias[0:len(kValues)])
plt.title("Elbow method")
# plt.savefig("./MLT/cwk/evaluation.png", dpi=fig.dpi)
plt.show()

