# user-based-recommender
User-based Collaborative Filtering using Pearson Correlation on MovieLens data
Project Overview:

This project implements a user-based collaborative filtering recommender system using Pearson Correlation Coefficient (PCC). It analyzes user similarity based on shared movie ratings and predicts new ratings for unseen movies.

The system uses MovieLens-style CSV datasets (ratings.csv and movies.csv) and suggests the top N recommended movies for a selected target user.

Features:

Builds a user-item matrix using pandas

Calculates Pearson correlation between users who rated at least k common movies

Predicts ratings based on the top N most similar users

Recommends movies the user hasn't seen, based on predicted ratings

Works with real MovieLens-type data

Technologies Used:

Python

pandas

scipy.stats (for Pearson correlation)

CSV files (ratings.csv, movies.csv)

How to Run:

Make sure you have Python 3 installed

Install the required packages:
pip install pandas scipy

Place recommender.py, ratings.csv, and movies.csv in the same folder

Run the script:
python recommender.py

Script Output Example:

Top users similar to user 1 (k=5):
User 17 -> PCC: 0.89
User 34 -> PCC: 0.85
...

Predicted rating for user 1 on movie 120 is: 4.25

Top 10 recommended movies for user 1:
The Matrix -> Predicted Rating: 4.78
Fight Club -> Predicted Rating: 4.70
...

Folder Structure:

user-based-recommender/
├── recommender.py (your main script)
├── ratings.csv (user ratings)
├── movies.csv (movie titles and IDs)
└── README.txt (this file)

