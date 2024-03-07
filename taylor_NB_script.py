import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# load in dataset
taylor = pd.read_csv("allTaylorLyrics_2024.csv", usecols=[1, 2, 3, 4, 5])

taylor.loc[(taylor['track_name'] == "All Too Well (10 Minute Version) (Taylor's Version) (From The Vault)"), 'track_name'] = "All Too Well 10"
taylor["track_name"] = taylor["track_name"].str.replace(r'\s*\([^)]*\)', '', regex=True)

# train-test split
lyrics_train, lyrics_test, song_train, song_test = train_test_split(taylor["lyric"], 
                                                                    taylor["track_name"], 
                                                                    test_size = 0.2, random_state = 21)


# load in the best model
with open("taylor_naive_bayes.pickle", 'rb') as m:
    model = pickle.load(m)

# load in the associated vectorizer
with open("taylor_NB_vectorizer.pickle", "rb") as v: 
    vectorizer = pickle.load(v)
    

def predict_song(lyric, model=model, vectorizer=vectorizer): 
    vec = vectorizer.transform([lyric])
    song_prediction = model.predict(vec)
    classes = model.classes_
    top_15 = sorted(zip(model.predict_proba(vec)[0], classes), reverse=True)[:15]
    return (song_prediction, top_15)


def check_songs(model=model, vectorizer=vectorizer, n=5, test_lyrics=lyrics_test, test_songs=song_test, random_state=21): 
    for lyric, correct_song in zip(test_lyrics.sample(n, random_state=random_state), test_songs.sample(n, random_state=random_state).values): 
        song_prediction, top_15 = predict_song(model, vectorizer, lyric)
        print("Lyric: ", lyric, "\n", 
              "Predicted song: ", song_prediction, "\n", 
              "Correct song: ", correct_song, "\n",
              "Top 15: ", top_15, "\n\n", sep="")  
         
        
