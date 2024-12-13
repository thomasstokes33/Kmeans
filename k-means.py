#Some code from NLTK.ORG

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


def clearFile(filename):
    open(filename,"w").close()

def writeToFile(filename, value):
    with open(filename, "a+") as file:
        file.write(str(value) + ",")
        
MINIMUM_WORD_FREQUENCY = 10
WINDOW_SIZE = 2 #How many words in sequence to consider to be in the window (either side)
# stop_words = set(stopwords.words('english')).union(set(stopwords.words('german'))).union(stopwords.words('spanish')).union(stopwords.words('french'))
# filename = "MLT/cwk/text8"
# raw = open(filename).read()

# print("type of raw:", type(raw)) #what is the type of the variable
# print("raw length", len(raw)) #what is the length of the text file, number of words
# tokenizer = RegexpTokenizer(r'[a-z]+')
# tokens = tokenizer.tokenize(raw)
# print("tokens type:", type(tokens))
# words = [w.lower() for w in tokens if not w.lower() in stop_words] # removes doesn and t and s, and words such as the.
# print("size (stop words removed):", len(words))
# print("Lemmatizing")
# lemmatizer = WordNetLemmatizer()
# words = [ lemmatizer.lemmatize(w) for w in words ]
# output = open("./MLT/cwk/lemmatized.txt", "w")
# output.write(" ".join(words))
# output.close()
print("Hyperparameters are")
print(" Window Size:", WINDOW_SIZE)
print(" Frequency boundaries:", MINIMUM_WORD_FREQUENCY)
filename = "MLT/cwk/lemmatized.txt"
raw = open(filename).read()
tokenizer = RegexpTokenizer(r'[a-z]+')
tokens = tokenizer.tokenize(raw)
words = tokens
frequency = defaultdict(int)
for word in words:
    frequency[word] +=1
newWords = [w for w in words if frequency[w] >= MINIMUM_WORD_FREQUENCY  and len(w) > 2]
# Create a list of unique words
print("words size" , len(words))

#https://www.geeksforgeeks.org/co-occurence-matrix-in-nlp/
#Let's build cooccurrence counts
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter) # creates dict with default value as a Counter.
newWordsSet = set(newWords)
print("Length of new words:", len(newWords))
for i, word in enumerate(words):
    if word in newWordsSet:
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)):
            if i != j and words[j] in newWordsSet:
                co_occurrences[word][words[j]] += 1
unique_words = list(set(newWords))
print ("unique words size", len(unique_words))

# co_matrix = csr_matrix((len(unique_words), len(unique_words)), dtype=np.int16)
co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=np.int16)

word_index = {word: idx for idx, word in enumerate(unique_words)}
for word, neighbors in co_occurrences.items():
    for neighbor, count in neighbors.items():
        co_matrix[word_index[word]][word_index[neighbor]] = np.array(count).astype(np.int16)

# Create a DataFrame for better readability
co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
co_matrix_df.to_pickle("./MLT/cwk/co_matrix.pkl")
print("Splitting into training and validation")
train_validation_set, test_set = train_test_split(co_matrix_df, test_size=0.2, random_state=7)
train_set, validation_set = train_test_split(train_validation_set, test_size=0.2, random_state=7)
silhoutteScores = []
inertias = []
for k in kValues:
    print("Trying with num clusters =", k)
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(train_set)
    labels = km.labels_
    print(set(labels))
    new_labels = km.predict(validation_set)
    inertia = km.inertia_
    if (len(set(new_labels))>= 2):
        silhouette_val = silhouette_score(validation_set, new_labels)
        print("Silhouette score", silhouette_val)
        silhoutteScores.append(silhouette_val)
        inertias.append(inertia)
    else:
        print("Infufficient number of labels")
plt.subplot(1, 2, 1)
plt.plot(kValues,silhoutteScores)
plt.title("Silhoutte scores")
plt.xlabel("K value")
plt.ylabel("Silhoutte score")
plt.subplot(1, 2, 2)
plt.xlabel("K value")
plt.ylabel("Inertia")
# plt.scatter(validation_set.iloc[:, 0], validation_set.iloc[:, 1], c=new_labels) # selects column 0 (all rows) as x coords and column 1 (all rows) as y. and then cluster labels.
plt.plot(kValues, inertias)
plt.title("Elbow method")
plt.show()