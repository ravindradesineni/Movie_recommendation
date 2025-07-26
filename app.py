import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# --- Load data ---
movies = pd.read_csv("movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("ratings_small.csv")

# --- Clean movies data ---
movies = movies[movies['id'].apply(lambda x: str(x).isdigit())]
movies['id'] = movies['id'].astype(int)
movies = movies[['id', 'title', 'overview']]
movies['overview'] = movies['overview'].fillna('')

# --- Merge datasets ---
ratings['movieId'] = ratings['movieId'].astype(int)
merged = pd.merge(ratings, movies, left_on='movieId', right_on='id')

# --- Create pivot table (user-item matrix) ---
user_movie_matrix = merged.pivot_table(index='userId', columns='title', values='rating')
movie_similarity = cosine_similarity(user_movie_matrix.fillna(0).T)

# --- Map movie titles to index ---
movie_list = user_movie_matrix.columns.tolist()
title_to_index = {title: i for i, title in enumerate(movie_list)}

# --- Recommendation function (Collaborative Filtering) ---
def get_similar_movies_cf(selected_movie, top_n=5):
    if selected_movie not in title_to_index:
        return ["Movie not found in ratings."]
    
    idx = title_to_index[selected_movie]
    sim_scores = list(enumerate(movie_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_movie_indices = [i[0] for i in sim_scores]
    return [movie_list[i] for i in similar_movie_indices]

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System (CF + Metadata)")

movie_input = st.selectbox("Pick a movie you like", sorted(movie_list))

if st.button("Recommend"):
    st.subheader("Top 5 Similar Movies (Item-Based CF):")
    recommendations = get_similar_movies_cf(movie_input)
    for title in recommendations:
        st.write("👉", title)
