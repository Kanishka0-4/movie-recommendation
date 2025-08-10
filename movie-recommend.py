import os
import math
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

OMDB_API_KEY = "4414dd0"  # Your OMDb key
TOP_K_NEIGHBORS = 20

movies = pd.read_csv(
    "ml-100k/u.item", sep="|", encoding="latin-1", header=None,
    names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
           'unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
           'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],
    usecols=[0, 1] + list(range(5, 24))
)
movies["movie_id"] = movies["movie_id"].astype(int)
movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip().str.lower()

ratings = pd.read_csv("ml-100k/u.data", sep="\t", header=None,
                      names=["user_id", "movie_id", "rating", "timestamp"])

GENRE_COLS = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
              'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

user_item = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating')

user_means = train_matrix.mean(axis=1)
centered = train_matrix.sub(user_means, axis=0).fillna(0)
user_similarity = cosine_similarity(centered)
user_similarity_df = pd.DataFrame(user_similarity, index=centered.index, columns=centered.index)

genre_matrix = movies.set_index('movie_id')[GENRE_COLS].fillna(0)
genre_sim = cosine_similarity(genre_matrix.values)
genre_index_to_movieid = list(genre_matrix.index)

def predict_rating_user_based(user_id, movie_id, k=TOP_K_NEIGHBORS):
    if user_id not in train_matrix.index:
        return train_df["rating"].mean()
    if movie_id not in train_matrix.columns:
        return user_means.get(user_id, train_df["rating"].mean())

    sims = user_similarity_df.loc[user_id]
    users_who_rated = train_matrix[train_matrix[movie_id].notna()].index
    if len(users_who_rated) == 0:
        return user_means.get(user_id, train_df["rating"].mean())

    cand_sims = sims.loc[users_who_rated].drop(index=user_id, errors="ignore")
    top_neighbors = cand_sims.abs().sort_values(ascending=False).head(k).index

    numerator, denominator = 0.0, 0.0
    for v in top_neighbors:
        sim = sims[v]
        r_vi = train_matrix.at[v, movie_id]
        if pd.isna(r_vi):
            continue
        mean_v = user_means.get(v, train_df["rating"].mean())
        numerator += sim * (r_vi - mean_v)
        denominator += abs(sim)

    if denominator == 0:
        return user_means.get(user_id, train_df["rating"].mean())

    pred = user_means.loc[user_id] + numerator / denominator
    return max(1.0, min(5.0, pred))

def fetch_omdb(title):
    try:
        r = requests.get("http://www.omdbapi.com/", params={"apikey": OMDB_API_KEY, "t": title}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}

def recommend_by_title(query_title, top_n=5):
    matches = get_close_matches(query_title.lower(), movies["clean_title"].tolist(), n=1, cutoff=0.4)
    if not matches:
        print("No movie found.")
        return
    matched_movie = movies[movies["clean_title"] == matches[0]].iloc[0]
    movie_id = matched_movie["movie_id"]

    idx = genre_index_to_movieid.index(movie_id)
    sim_scores = list(enumerate(genre_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_ids = [genre_index_to_movieid[i] for i, _ in sim_scores]

    for mid in similar_ids:
        title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
        pred_rating = predict_rating_user_based(1, mid)  # User ID fixed for demo
        omdb = fetch_omdb(title)
        print(f"{title} | Pred: {pred_rating:.2f} | IMDb: {omdb.get('imdbRating', 'N/A')}")
        print(f"Plot: {omdb.get('Plot', '')[:100]}...")
        print("-" * 40)


movie = input("Enter a movie title: ")
recommend_by_title(movie)
