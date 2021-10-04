"""
Sina Dabiri

LAB/HOMEWORK.
BMI 500; WEEK 6: NATURAL LANGUAGE PROCESSING 2

1. You are provided two text files (text1 and text2)
a) Preprocess the two texts by lowercasing them.

"""

text1 = open("text1", "r", encoding="utf8").read().lower()
text2 = open("text2", "r", encoding="utf8").read().lower()
print("The first file's content are: \n", text1, '\n\n', "The second file's content are: \n", text2)

'''
b) Follow the instructions on https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
to vectorize the two texts using the CountVectorizer. Use n-gram size of 1-3 for the vectorizer.
'''
# First step: generate vocabulary
# Second step: vectorize

from sklearn.feature_extraction.text import CountVectorizer
corpus = [text1, text2]
# CountVectorizer coverts the collection of text, corpus, into a matrix of token counts
vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorized_corpus = vectorizer.fit_transform(corpus)

'''
c) Compute the cosine similarity between the texts (sklearn provides a function for it).
'''
from sklearn.metrics.pairwise import cosine_similarity
vect1 = vectorized_corpus[0]
vect2 = vectorized_corpus[1]
cos_similarity = cosine_similarity(vect1, vect2)

print("The cosine similarity is: ", cos_similarity)

'''
d) Compute the jaccard similarity between text1 and text2
'''
from sklearn.metrics import jaccard_score
import numpy as np
# print(np.shape(vect1), '\n', np.shape(vect2))
# print(np.dtype(vect1), '\n', np.dtype(vect2))

# remove white space, tokenize
text1_token = text1.strip().split()
text2_token = text2.strip().split()
text1_set = set(text1_token)
text2_set = set(text2_token)
# print("Tokenized text1 as a set is: ", text1_set)
# print("Tokenized text2 as a set is: ", text2_set)

# Find the intersection of the two vectors
intersec = text1_set.intersection(text2_set)

# Find the union of the two vectors
union = text1_set.union(text2_set)

# Calculate Jaccard similarity
jaccard_similarity = float(len(intersec)/len(union))
print("The Jaccard similarity is: ", jaccard_similarity)

'''
2. Read section 4 and 5 of the nltk book: https://www.nltk.org/book/ch05.html.
a) Follow the instructions to train a POS tagger.
Show the performance of different taggers (e.g., unigram, bigram and combined taggers) on your chosen corpus (e.g. the brown corpus).
Use 90% of the data for training and 10% for evaluation.
Compare the performances of at least 3 taggers.
Save and load the tagger.
'''

import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
# tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
# conditional_fd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
# print("The conditional Freq Distribution is: ", conditional_fd)

# Unigram tagging
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print("Accuracy of Unigram Tagger: ", unigram_tagger.evaluate(test_sents))

# Bigram tagging
bigram_tagger = nltk.BigramTagger(train_sents)
print("Accuracy of Bigram Tagger: ", bigram_tagger.evaluate(test_sents))

# Combined Taggers
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, cutoff=2, backoff=t0)
t2 = nltk.BigramTagger(train_sents, cutoff=2, backoff=t1)
t3 = nltk.TrigramTagger(train_sents, cutoff=2, backoff=t2)
print("Accuracy of Combined Tagger: ", t3.evaluate(test_sents))
# Compare performances
# Bigram accuracy is significantly lower due to sparse data issue for context information.
'''The combined method uses Trigram first, if didn't find tag tries Bigram, if failed uses Unigram, 
if failed then uses the default tagger of 'NN'. '''

# Save and load tagger
from pickle import dump
output = open('t3.pkl', 'wb')
dump(t3, output, -1)
output.close()

from pickle import load
input = open('t3.pkl', 'rb')
tagger = load(input)
input.close()
'''
Use a trained tagger to tag all the words from text1.
Can lowercasing affect the performance of the POS tagger? (Put your answer as comment in python file)
'''
tokens = text1_token
print("The Combo Tagger tagging of text1 is:", tagger.tag(tokens))
# Lowercasing improves performance of a tagger by not creating different categories of a word with different casings.

'''
3. What are some of the possible NLP components of a Question Answering System?
'''

# 1- Query Analysis, where subdomain of the question is identified.
# 2- Single-doc summary, where parts of a document relevant to the query is identified.
# 3- Multi-doc summary, where relevant parts of multiple documents are combined.
# 4- Quality assessment, where QC is performed. i.e. in Medical filed whether evidence came from Randomized Controlled
#   trials or case report. 
