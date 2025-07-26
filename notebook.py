# 1. Imports
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 2. Load Data
ratings = pd.read_csv('ratings_small.csv')  # Path in Colab
movies = pd.read_csv('movies_metadata.csv', low_memory=False)

# 3. Clean and Filter
# Convert IDs to numeric (some movie_ids in movies_metadata are not clean)
movies = movies[movies['id'].apply(lambda x: x.isnumeric())]
movies['id'] = movies['id'].astype(int)

# Merge both datasets on movieId
merged = pd.merge(ratings, movies, left_on='movieId', right_on='id')

# Drop NaN and duplicates
merged = merged[['userId', 'title', 'genres', 'rating']].dropna().drop_duplicates()

# 4. Content-Based Filtering Setup
# Convert genres from stringified lists to clean text
import ast

def clean_genres(g):
    try:
        g_list = ast.literal_eval(g)
        return ' '.join([i['name'] for i in g_list])
    except:
        return ''

merged['clean_genres'] = merged['genres'].apply(clean_genres)

# Create movie-genre matrix using CountVectorizer
cv = CountVectorizer(stop_words='english')
genre_matrix = cv.fit_transform(merged.drop_duplicates(subset='title')['clean_genres'])

# Cosine similarity between all movies
cos_sim = cosine_similarity(genre_matrix)

# Index mapping
movie_titles = merged.drop_duplicates(subset='title')['title'].reset_index(drop=True)
movie_indices = pd.Series(movie_titles.index, index=movie_titles)

# 5. Function: Recommend based on movie title
def recommend_by_content(title, top_n=5):
    if title not in movie_indices:
        return ["Movie not found."]
    idx = movie_indices[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movie_titles[i[0]] for i in sim_scores[1:top_n+1]]
    return top_movies

# 6. Collaborative Filtering Setup (user-based)
# Create pivot table (users x movies)
user_movie_matrix = merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute user similarity
user_sim = cosine_similarity(user_movie_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# 7. Function: Recommend based on similar users
def recommend_by_user(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return ["User not found."]
    # Get similar users
    sim_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    
    # Weighted average rating of top-N similar users
    sim_movies = user_movie_matrix.loc[sim_users.index]
    weighted_ratings = sim_movies.T.dot(sim_users) / sim_users.sum()
    
    # Filter out movies already rated by the user
    user_seen = user_movie_matrix.loc[user_id]
    unseen_movies = weighted_ratings[user_seen == 0]
    
    return unseen_movies.sort_values(ascending=False).head(top_n).index.tolist()

# 8. Example Usage
print("Content-based recommendations for 'Toy Story':")
print(recommend_by_content('The Man with the Golden Arm'))

print("\nCollaborative filtering recommendations for user ID 1:")
print(recommend_by_user(1))
