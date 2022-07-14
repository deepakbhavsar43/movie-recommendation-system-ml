import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data_path = 'dataset/toy_dataset.csv'
df = pd.read_csv(data_path, index_col=0)
df = df.fillna(0)


def standardize(row):
    std_row = (row - row.mean()) / row.max() - row.min()
    return std_row


std_df = df.apply(standardize)
item_similarity = cosine_similarity(std_df.T)
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)


def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name] * (user_rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


print(get_similar_movies("romantic1", 2))
