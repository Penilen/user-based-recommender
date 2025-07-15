import pandas as pd
from scipy.stats import pearsonr

# Load the CSV files
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Preview the content
print("Ratings preview:")
print(ratings.head())

print("\nMovies preview:")
print(movies.head())

# Create the user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
print("\nUser-Item Rating Matrix:")
print(user_item_matrix.head())

# Step 2 - Manual PCC computation with k threshold
target_user = 1      # can be changed to any userId
k = 5                # minimum number of shared movies

# Get the ratings of the target user
u_ratings = user_item_matrix.loc[target_user]

# Dictionary to store results
similarities = {}

# Compare with all other users
for other_user in user_item_matrix.index:
    if other_user == target_user:
        continue  # skip comparing the user to themselves

    # Get the ratings of the other user
    v_ratings = user_item_matrix.loc[other_user]

    # Find shared movies (non-NaN in both)
    common = u_ratings.notna() & v_ratings.notna()
    shared_count = common.sum()

    if shared_count >= k:
        # Extract only shared ratings
        u_common = u_ratings[common]
        v_common = v_ratings[common]

        # Compute Pearson correlation
        corr, _ = pearsonr(u_common, v_common)
        similarities[other_user] = corr

# Show sorted list of users by similarity to user 1
sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop users similar to user {target_user} (k={k}):")
for user, score in sorted_similarities[:10]:
    print(f"User {user} -> PCC: {score:.4f}")

def predict_rating(user_id, movie_id, similarities, user_item_matrix, n=5):
    # Get the average rating of the target user
    u_ratings = user_item_matrix.loc[user_id]
    mu_u = u_ratings.mean()

    # Get top-n similar users sorted by PCC
    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    numerator = 0
    denominator = 0
    count = 0

    for v_id, pcc in similar_users:
        v_ratings = user_item_matrix.loc[v_id]
        rvp = v_ratings.get(movie_id)

        if pd.notna(rvp):  # Only use users who rated movie p
            mu_v = v_ratings.mean()
            numerator += pcc * (rvp - mu_v)
            denominator += abs(pcc)
            count += 1

        if count >= n:
            break

    if denominator == 0:
        return mu_u  # fallback to user's average rating
    return mu_u + (numerator / denominator)

# Step 3 - Predict a rating
target_movie = 120  # pick a movie the user hasn't rated
n = 5               # number of similar users to use

predicted = predict_rating(target_user, target_movie, similarities, user_item_matrix, n=n)
print(f"\nPredicted rating for user {target_user} on movie {target_movie} is: {predicted:.2f}")

# # Step 4: Recommend Top-N movies to the target user
# N = 5  # Number of movies to recommend
# predictions = {}

# # Go through all movies
# for movie_id in user_item_matrix.columns:
#     if pd.isna(user_item_matrix.loc[target_user, movie_id]):
#         # Only predict for movies the user hasn't rated
#         predicted_rating = predict_rating(target_user, movie_id, similarities, user_item_matrix, n=5)
#         predictions[movie_id] = predicted_rating


# Step 4 - Recommend top movies for the user (based on predicted ratings)
recommendations = {}

unrated_movies = [mid for mid in user_item_matrix.columns if pd.isna(user_item_matrix.loc[target_user, mid])]
unrated_movies = unrated_movies[:100]  # limit for testing

for movie_id in unrated_movies:
    pred = predict_rating(target_user, movie_id, similarities, user_item_matrix, n=5)
    recommendations[movie_id] = pred

# Show top recommended movies
top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 recommended movies for user {target_user}:")
for movie_id, predicted_rating in top_recommendations:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"{title} -> Predicted Rating: {predicted_rating:.2f}")


