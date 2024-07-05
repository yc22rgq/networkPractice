import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.setrecursionlimit(10000000)
df = pd.read_csv('books.csv', nrows=200)

df.fillna('', inplace=True)

df['text'] = df['title'] + ' ' + df['authors'] + ' ' + df['language_code']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])

knn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn.fit(tfidf_matrix)


def get_recommendations(title):
    title_vector = vectorizer.transform([title])

    distances, indices = knn.kneighbors(title_vector)

    return df['title'].iloc[indices[0]].tolist()


recommendations = get_recommendations('La tía Julia y el escribidor')
for rec in recommendations:
    print(rec)