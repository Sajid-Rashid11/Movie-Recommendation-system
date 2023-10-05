import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.stem.porter import PorterStemmer

'''Here we extract the data from the csv files and select the columns needed.'''
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview',
                'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)


ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)


def convert_to_list(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


def selectTop4(obj):
    c = 0
    l = []
    for i in ast.literal_eval(obj):
        if c != 4:
            l.append(i['name'])
            c += 1
        else:
            break
    return l


def getDirector(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break

    return l


'''Here we process the data by changing the string values in arrays containing only the desired info such as top 4 cast name,
 just director name, convering the overview into an array of words, etc..'''
movies['genres'] = movies['genres'].apply(convert_to_list)
movies['keywords'] = movies['keywords'].apply(convert_to_list)
movies['cast'] = movies['cast'].apply(selectTop4)
movies['crew'] = movies['crew'].apply(getDirector)
movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(' ', '') for i in x])
movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(' ', '') for i in x])

"""these for loops were made just for some personal insight of the structure of the datframe. 
Normally lambda function is faster and better for large datasets."""
for i in movies.index:
    movies['overview'][i] = movies['overview'][i].split(' ')

for i in movies['crew'].index:
    if len(movies['crew'][i]) > 0:
        movies['crew'][i][0] = movies['crew'][i][0].replace(' ', '')

"""create a tag column and process it."""
movies['tags'] = movies['overview']+movies['genres'] + \
    movies['keywords']+movies['cast']+movies['crew']

"""creating a new dataframe with only tags column containg the data
and processinf it."""
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

"""count vectorizer keeps track of the most commonly occuring words in the concated tag list.
Here we have taken the top 5000 most common words and made a vector which keeps a word count for 
the no. of times the word appears in the movie. Stop word removes common english articles. This 
is Bag of Words (BoW) method."""
cv = CountVectorizer(max_features=5000, stop_words='english')
new_df['tags'] = new_df['tags'].apply(stem)
vectors = cv.fit_transform(new_df['tags']).toarray()

tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.a
