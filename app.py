# app.py
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Genre columns
genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

#load data
@st.cache_data
def load_data():
    movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', header=None,
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols,
                         usecols=[0, 1, 2] + list(range(5, 24)))
    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')[0]
    fallback_year = movies['release_date'].str.extract(r'(\d{4})')[0]
    movies['release_year'] = movies['release_year'].fillna(fallback_year)
    return movies

@st.cache_data
def load_ratings():
    return pd.read_csv("ml-100k/u.data", sep='\t', header=None,
                       names=['user_id', 'movie_id', 'rating', 'timestamp'])

movies = load_data()
ratings = load_ratings()

title_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()
genre_features = movies[genre_cols]
cosine_sim = cosine_similarity(genre_features)

# recommendation functions
@st.cache_data
def get_recommendations(user_input):
    user_input = user_input.strip().lower()

    movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.lower().str.strip()

    matched = movies[movies['clean_title'].str.contains(user_input)]

    if matched.empty:
        return ["❌ Movie not found."]
    
    idx = matched.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    top_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[top_indices].tolist()


def search_movies(query):
    query = query.strip().lower()
    if not query:
        return ["⚠️ Please enter a search term."]

    results = pd.DataFrame()

    # year
    if query.isdigit() and len(query) == 4:
        results = movies[movies['release_year'] == query]

    # genre
    elif query.capitalize() in genre_cols:
        results = movies[movies[query.capitalize()] == 1]

    # keyword in title
    else:
        results = movies[movies['title'].str.lower().str.contains(query)]

    return results['title'].tolist()[:10] if not results.empty else ["No matches found."]

def top_movies_by_user(user_id):
    if user_id not in ratings['user_id'].values:
        return ["User ID not found."]
    merged = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
    user_data = merged[merged['user_id'] == user_id]
    top = user_data.sort_values(by='rating', ascending=False).drop_duplicates('title').head(5)
    return top[['title', 'rating']].values.tolist()

# streamlit

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")

st.caption("Enter movie titles, genres, or years — and get recommendations!")

# 🔹 Feature 1: Similar movies
st.subheader("🎯 Get Similar Movies")
movie_title = st.text_input("Enter a movie title (e.g. 'Star Wars')", key="movie_input")

if movie_title:
    recs = get_recommendations(movie_title)
    st.write("Top 5 Similar Movies:")
    for r in recs:
        st.markdown(f"👉 {r}")

# 🔹 Feature 2: Unified Search
st.subheader("🔍 Search Movies")
search_query = st.text_input("Type anything: e.g. 'comedy', 'love', '1995'", key="search_input")

if search_query:
    matched_movies = search_movies(search_query)
    st.write("Matching Movies:")
    for m in matched_movies:
        st.markdown(f"🎬 {m}")

# 🔹 Feature 3: User's Top Rated Movies
st.subheader("👤 Top Movies by User")
user_input = st.text_input("Enter User ID (1–943):", key="user_input")

if user_input and user_input.isdigit():
    top_user = top_movies_by_user(int(user_input))
    st.write("Top 5 Movies Rated by User:")
    for item in top_user:
        if isinstance(item, str):
            st.warning(item)
        else:
            st.markdown(f"⭐ {item[1]} — {item[0]}")
elif user_input and not user_input.isdigit():
    st.error("Please enter a valid numeric user ID.")

st.markdown("---")
st.caption("Powered by MovieLens 100k | Built with ❤️ using Streamlit")
