# This script preproccess a dataset and plots increasing values of K used for the K-Means model
# against the result of 4 evaluation methods on the model.

#Import NLTK used for tokenisation and stop words
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from scipy.stats import iqr

def clearFile(filename):
    open(filename,"w").close()

def writeToFile(filename, row):
    with open(filename, "w+", newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row)


def clearFiles():
    clearFile("./MLT/cwk/inertias.txt")
    clearFile("./MLT/cwk/silhouettes.txt")
    clearFile("./MLT/cwk/kValue.txt")
    clearFile("./MLT/cwk/davies.txt")
    clearFile("./MLT/cwk/calinski.txt")


kValues= [3,4,5,6,7,8,9, 10,11, 12, 15, 20, 25, 30]
MINIMUM_WORD_FREQUENCY = 20
WINDOW_SIZE = 2 # The number of words to compare with the current word in the sequence(either side).
stop_words = set(stopwords.words('english')).union(set(stopwords.words('german'))).union(stopwords.words('spanish')).union(stopwords.words('french')) # Create stopwords set for multiple languages
filename = "MLT/cwk/text8"
raw = open(filename).read()

print("raw length", len(raw))
tokenizer = RegexpTokenizer(r'[a-zA-Z]+[a-zA-Z]+') # Get words as tokens if longer than 2 characters
tokens = tokenizer.tokenize(raw)
print("tokens type:", type(tokens))
words = [w.lower() for w in tokens if not w.lower() in stop_words] # Remove doesn and t and s, and words such as the.
print("size (stop words removed):", len(words))
print("Lemmatizing")
lemmatizer = WordNetLemmatizer()
words = [ lemmatizer.lemmatize(w) for w in words ] # Lemmatize words so they are legitimate stems.
output = open("./MLT/cwk/lemmatized.txt", "w")
output.write(" ".join(words)) # Backup lemmatized words as that process is fairly slow.
output.close()
print("Hyperparameters are")
print(" Window Size:", WINDOW_SIZE)
print(" Frequency boundaries:", MINIMUM_WORD_FREQUENCY)
filename = "MLT/cwk/lemmatized.txt"
raw = open(filename).read()
tokenizer = RegexpTokenizer(r'[a-z]+[a-z]+')
tokens = tokenizer.tokenize(raw)
words = tokens
# Create a word frequency counter
frequency = defaultdict(int)
for word in words:
    frequency[word] +=1
newWords = [w for w in words if frequency[w] >= MINIMUM_WORD_FREQUENCY  and len(w) > 2] # Remove words with frequency that is less than MINIMUM_WORD_FREQUENCY
print("words size" , len(words))

co_occurrences = defaultdict(Counter) # Create dict with default value as a Counter.
newWordsSet = set(newWords)
print("Length of new words:", len(newWords))
# Inspired by code at https://www.geeksforgeeks.org/co-occurence-matrix-in-nlp/
# Iterate over the original words, counting co-occurences with words in the reduced word set, using a shifting window.
for i, word in enumerate(words):
    if word in newWordsSet:
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)): # Use a window to check before and after words.
            if i != j and words[j] in newWordsSet:
                co_occurrences[word][words[j]] += 1
unique_words = list(set(newWords))
print ("unique words size", len(unique_words))

co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=np.int16)

word_index = {word: idx for idx, word in enumerate(unique_words)} # Setup a dictionary mapping words to indexes in unique words.
# Create a co-occurrence matrix that is unique_words*unique_words in size where position ij is the number of times word i co-occurs with word j.
for word, neighbors in co_occurrences.items():
    for neighbor, count in neighbors.items():
        co_matrix[word_index[word]][word_index[neighbor]] = np.array(count).astype(np.int16)

# Create a DataFrame for better readability
co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
co_matrix_df.to_pickle("./MLT/cwk/co_matrix.pkl") # Write to Pickle file as backup.
co_matrix_df = pd.read_pickle("./MLT/cwk/co_matrix.pkl")
print("Splitting into training and validation")
# Split into a training, validation and final testing set.
train_validation_set, test_set = train_test_split(co_matrix_df, test_size=0.2, random_state=7)
train_set, validation_set = train_test_split(train_validation_set, test_size=0.2, random_state=7)
silhouetteScores = []
inertias = []
testedKValues = []
daviesValues = []
calinskiValues = []
clearFiles()
for k in kValues:
    print("Trying with num clusters =", k)
    tempSilhouettes = []
    tempInertias = []
    tempCalinski = []
    tempDavies = []
    for seed in range(42,51,4): # Try different random seeds for the K-means model.
        print("Seed:",seed)
        kf = KFold(n_splits=4)
        for train_index, test_index in kf.split(train_validation_set): # Cross validate using the training data as the validation data and vice versa.
            print(train_index, test_index)
            train_set, validation_set = train_validation_set.iloc[train_index], train_validation_set.iloc[test_index]
            km = KMeans(n_clusters=k, random_state=seed)
            km.fit(train_set)
            labels = km.labels_
            print(set(labels))
            new_labels = km.predict(validation_set)
            inertia = km.inertia_
            if (len(set(new_labels))>= 2): # Check the number of labels is enough to perform evaluation.
                silhouette_val = silhouette_score(validation_set, new_labels)
                print("Silhouette val", silhouette_val)
                tempSilhouettes.append(silhouette_val)
                tempInertias.append(inertia)
                tempDavies.append(davies_bouldin_score(validation_set, new_labels))
                tempCalinski.append(metrics.calinski_harabasz_score(validation_set, new_labels))
                
            else:
                print("Insufficient number of labels")
    # Remove outliers
    tempSilhouettes = np.array(tempSilhouettes)
    silhouetteIQR = iqr(tempSilhouettes)
    silhouetteQuartile1 = np.percentile(tempSilhouettes, 25)
    silhouetteQuartile3 = np.percentile(tempSilhouettes, 75)
    silhouetteScaledIQR = 1.5 * iqr(tempSilhouettes)
    indexesToDrop = np.where((tempSilhouettes < silhouetteQuartile1 - silhouetteScaledIQR) | (tempSilhouettes > silhouetteQuartile3 + silhouetteScaledIQR))[0]
    tempSilhouettes = np.delete(tempSilhouettes, indexesToDrop)
    tempInertias = np.delete(tempInertias, indexesToDrop)
    tempCalinski = np.delete(tempCalinski,indexesToDrop)
    tempDavies = np.delete(tempDavies, indexesToDrop)
    if len(tempSilhouettes) > 0: # If there are values left, average the evaluation scores, for a given k.
        averageInertia = sum(tempInertias)/len(tempInertias)
        averageSilhouette = sum(tempSilhouettes)/len(tempSilhouettes)
        averageCalinksi = sum(tempCalinski)/len(tempCalinski)
        averageDavies = sum(tempDavies)/len(tempDavies)
        print("Results for k =", k)
        print("Silhouette score", averageSilhouette)
        print("Inertia", averageInertia)
        print("Calinski Harabasz score", averageCalinksi)
        print("Davies Bouldin score", averageDavies)

        inertias.append(averageInertia)
        silhouetteScores.append(averageSilhouette)
        testedKValues.append(k)
        calinskiValues.append(averageCalinksi)
        daviesValues.append(averageDavies)
        # Write k and evaluation metrics to backup files.
        writeToFile("./MLT/cwk/silhouettes.txt", silhouetteScores)
        writeToFile("./MLT/cwk/inertias.txt", inertias)
        writeToFile("./MLT/cwk/kValue.txt", testedKValues)
        writeToFile("./MLT/cwk/davies.txt", daviesValues)
        writeToFile("./MLT/cwk/calinski.txt", calinskiValues)
    else:
        print("All outliers")

# Plot results
fig = plt.figure(figsize=(10,6))
plt.subplot(2, 2, 1)
plt.plot(testedKValues,silhouetteScores)
plt.title("Silhouette scores")
plt.xlabel("K value")
plt.ylabel("Silhouette score")
plt.subplot(2, 2, 2)
plt.plot(testedKValues, inertias)
plt.title("Elbow method")
plt.xlabel("K value")
plt.ylabel("Inertia")
plt.subplot(2,2,3)
plt.title("Davies Bouldin score")
plt.xlabel("K value")
plt.ylabel("Davies Bouldin score")
plt.plot(testedKValues, daviesValues)
plt.subplot(2,2,4)
plt.title("Calinski Harabasz score")
plt.xlabel("K value")
plt.ylabel("Calinski Harabasz score")
plt.plot(testedKValues, calinskiValues)
plt.savefig("./MLT/cwk/evaluation.png", dpi=fig.dpi)
plt.show()