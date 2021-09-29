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
vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorized_corpus = vectorizer.fit_transform(corpus)
# print(vectorized_corpus)

'''
c) Compute the cosine similarity between the texts (sklearn provides a function for it).
'''
from sklearn.metrics.pairwise import cosine_similarity
vect1 = vectorized_corpus[0]
vect2 = vectorized_corpus[1]
cos_similarity = cosine_similarity(vect1, vect2)

print(cos_similarity)
