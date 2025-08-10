from flask import Flask, render_template, request
import pandas as pd
import requests
import re
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

OMDB_API_KEY = "4414dd0"
CACHE_FILE = "omdb_cache.json"
GENRE_COLS = [
    'unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
    'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
]

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        omdb_cache = json.load(f)
else:
    omdb_cache = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(omdb_cache, f, indent=2, ensure_ascii=False)

movies = pd.read_csv(
    "ml-100k/u.item", sep="|", encoding="latin-1", header=None,
    names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + GENRE_COLS,
    usecols=[0, 1] + list(range(5, 5 + len(GENRE_COLS)))
)
movies["movie_id"] = movies["movie_id"].astype(int)
movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip().str.lower()

ratings = pd.read_csv("ml-100k/u.data", sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])
pop_counts = ratings.groupby("movie_id").size().to_dict()

genre_matrix = movies.set_index("movie_id")[GENRE_COLS].fillna(0).astype(int)
genre_sim = cosine_similarity(genre_matrix.values)
genre_index_to_movieid = list(genre_matrix.index)

def clean_title_for_api(title: str) -> str:
    return re.sub(r"\(\d{4}\)", "", str(title)).strip()

def fetch_omdb_exact(title: str):
    key = clean_title_for_api(title)
    if key in omdb_cache:
        return omdb_cache[key]

    try:
        r = requests.get("http://www.omdbapi.com/", params={"apikey": OMDB_API_KEY, "t": key, "type": "movie"}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data.get("Response") == "True":
                omdb_cache[key] = data
                save_cache()
                return data
    except Exception:
        pass

    omdb_cache[key] = {}
    save_cache()
    return {}

def fetch_omdb_search_fallback(query: str):
    exact = fetch_omdb_exact(query)
    if exact:
        return exact

    try:
        r = requests.get("http://www.omdbapi.com/", params={"apikey": OMDB_API_KEY, "s": query, "type": "movie"}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data.get("Response") == "True" and data.get("Search"):
                first = data["Search"][0]
                title = first.get("Title")
                if title:
                    return fetch_omdb_exact(title)
    except Exception:
        pass

    return {}

def token(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

SYNONYMS = {
    "sciencefiction": "Sci-Fi",
    "sciencefictionfilm": "Sci-Fi",
    "scifi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "family": "Children",
    "animation": "Animation",
    "animated": "Animation",
    "romcom": "Romance",
    "romantic": "Romance",
    "documentary": "Documentary",
    "filmnoir": "Film-Noir",
    "musical": "Musical",
    "mystery": "Mystery",
    "crime": "Crime",
    "action": "Action",
    "adventure": "Adventure",
    "horror": "Horror",
    "fantasy": "Fantasy",
    "drama": "Drama",
    "comedy": "Comedy",
    "thriller": "Thriller",
    "western": "Western",
    "war": "War"
}

def omdb_genres_to_vector(genre_str: str):
    vec = [0] * len(GENRE_COLS)
    if not genre_str:
        return vec
    omdb_tokens = [token(g) for g in genre_str.split(",")]

    for i, col in enumerate(GENRE_COLS):
        col_tok = token(col)
        for ot in omdb_tokens:
            if not ot:
                continue
            mapped = SYNONYMS.get(ot)
            if mapped and mapped.lower() == col.lower():
                vec[i] = 1
                break
            if ot in col_tok or col_tok in ot:
                vec[i] = 1
                break
    return vec

def find_best_seed_from_omdb_genres(omdb_genre_str: str):
    omdb_vec = omdb_genres_to_vector(omdb_genre_str)
    if sum(omdb_vec) == 0:
        return None

    overlap = genre_matrix.dot(pd.Series(omdb_vec, index=GENRE_COLS))
    best_id = None
    best_score = -1
    for mid, score in overlap.items():
        if score > best_score:
            best_score = int(score)
            best_id = mid
        elif score == best_score and score > 0:
            if pop_counts.get(mid, 0) > pop_counts.get(best_id, 0):
                best_id = mid

    if best_score <= 0:
        return None
    return int(best_id)

def get_similar_movies(query, top_n=12):
    q = str(query).strip().lower()
    if not q:
        return []

    matched = movies[movies["clean_title"].str.contains(q, na=False)]

    results = []
    for _, row in matched.iterrows():
        omdb_data = fetch_omdb_exact(row["title"])
        if omdb_data and omdb_data.get("Response") == "True":
            if (omdb_data.get("Poster") not in ("", "N/A") and
                omdb_data.get("Plot") not in ("", "N/A") and
                omdb_data.get("imdbRating") not in ("", "N/A")):
                results.append({
                    "title": row["title"],
                    "poster": omdb_data.get("Poster"),
                    "plot": omdb_data.get("Plot"),
                    "imdb": omdb_data.get("imdbRating")
                })
        if len(results) >= top_n:
            break

    return results

@app.route("/", methods=["GET", "POST"])
def home():
    recs = []
    heading = ""
    if request.method == "POST":
        q = request.form.get("movie", "")
        recs = get_similar_movies(q, top_n=12)
        heading = f"Search results for '{q}'" if q else ""
    return render_template("index.html", recs=recs, heading=heading)

if __name__ == "__main__":
    app.run(debug=True)
