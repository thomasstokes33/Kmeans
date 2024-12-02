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
# stop_words = set(stopwords.words('english')).union(set(stopwords.words('german'))).union(stopwords.words('spanish')).union(stopwords.words('french'))
# englishWords = set(nltk.corpus.words.words())
# filename = "MLT/cwk/text8"
# raw = open(filename).read()

# print("type of raw:", type(raw)) #what is the type of the variable
# print("raw length", len(raw)) #what is the length of the text file, number of words
# tokenizer = RegexpTokenizer(r'[a-z]+[a-z]+')
# tokens = tokenizer.tokenize(raw)
# print("tokens type:", type(tokens))
# words = [w.lower() for w in tokens if not w.lower() in stop_words] # removes doesn and t and s, and words such as the.
# # words = [w for w in words if w in englishWords]
# print("size (stop words removed):", len(words))

# lemmatizer = WordNetLemmatizer()
# words = [ lemmatizer.lemmatize(w) for w in words ]
# output = open("MLT/cwk/lematized.txt", "w")
# output.write(" ".join(words))
# output.close()

filename = "MLT/cwk/lemmatized.txt"
raw = open(filename).read()
tokenizer = RegexpTokenizer(r'[a-z]+[a-z]+')
tokens = tokenizer.tokenize(raw)
words = tokens
frequency = defaultdict(int)
for word in words:
    frequency[word] +=1
words = [w for w in words if frequency[w] >= 10 and frequency[w] < 5000 and len(word) > 2]
# Create a list of unique words
print("words size" , len(words))

#https://www.geeksforgeeks.org/co-occurence-matrix-in-nlp/
#Let's build cooccurrence counts
window_size = 2 #How many words in sequence to consider to be in the window (either side)
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter) # creates dict with default value as a Counter.
for i, word in enumerate(words):
    for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
        if i != j:
            co_occurrences[word][words[j]] += 1
unique_words = list(set(words))
print ("unique words size", len(unique_words))

# co_matrix = csr_matrix((len(unique_words), len(unique_words)), dtype=np.int16)
co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=np.int16)

word_index = {word: idx for idx, word in enumerate(unique_words)}
for word, neighbors in co_occurrences.items():
    for neighbor, count in neighbors.items():
        co_matrix[word_index[word]][word_index[neighbor]] = np.array(count).astype(np.int16)

# Create a DataFrame for better readability
co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
train_validation_set, test_set = train_test_split(co_matrix_df, test_size=0.2, random_state=7)
train_set, validation_set = train_test_split(train_validation_set, test_size=0.2, random_state=7)
km = KMeans(n_clusters=5)
km.fit(train_set)
labels = km.labels_
print(set(labels))
new_labels = km.predict(validation_set)
silhouette_avg = silhouette_score(validation_set, new_labels)
print("Silhouette av", silhouette_avg)

#TODO: WHAT IF WE USED A SYMMETRIC MATRIX