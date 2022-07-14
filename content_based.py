import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read csv file
dataset_path = 'dataset/movie_dataset.csv'
dataset = pd.read_csv(dataset_path)

# Select features
features = ["keywords", "cast", "genres", "director"]

# Fill missing values with blank space " "
dataset = dataset.fillna(" ")

# combine all the features and save it to a separate column
def combine_features(row):
    """
    :param row: It is row from the dataframe
    :return: returns the concatenation of desired features values from given row
    """
    return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]


dataset["combined_features"] = dataset.apply(combine_features, axis=1)
# count matrix for dataset["combined_features"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(dataset["combined_features"])

# cosine similarity based on count matrix
cosine_sim = cosine_similarity(count_matrix)


# functions to get title from index and index from title
def get_title_from_index(index):
    return dataset[dataset.index == index]["title"].values[0]


def get_index_from_title(title):
    return dataset[dataset.title == title]["index"].values[0]


# get index from movie title
movie_name = "Avatar"
movie_index = get_index_from_title(movie_name)
print(cosine_sim)
print("index: ", movie_index)
print("row on index : ", cosine_sim[movie_index])
exit()
# finding similar movies in context to movie name in variable movie_name
similar_movies = list(enumerate(cosine_sim[movie_index]))

# sorting similar movies in descending order based on their similarity with the movie in variable movie_name
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# print name of first 50 similar movies using their index
for movie in sorted_similar_movies[0:50]:
    print(get_title_from_index(movie[0]))

