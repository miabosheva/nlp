import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from gensim.models import Word2Vec
from scipy.stats import kendalltau, pearsonr

if __name__ == "__main__":
    # 3. A)
    df = pd.read_csv("combined.csv")

    model_word2vec = Word2Vec.load("model_word2vec.bin")

    def convert_to_word2vec(word):
        try:
            vector = model_word2vec.wv[word]
            return vector
        except KeyError:
            return None

    df['word2vec_word_1'] = df['Word 1'].apply(lambda x: convert_to_word2vec(x))
    df['word2vec_word_2'] = df['Word 2'].apply(lambda x: convert_to_word2vec(x))
    df = df.dropna()

    vectors_list1 = df['word2vec_word_1']
    vectors_list2 = df['word2vec_word_2']
    means = df['Human (mean)']

    # print(len(means))
    # print(len(euclidean_distance_scores))
    # print(len(cosine_similarity_scores))

    cosine_similarity_scores = []
    euclidean_distance_scores = []

    for vector1, vector2 in zip(vectors_list1, vectors_list2):
        similarity_score = cosine_similarity([vector1], [vector2])[0][0]
        cosine_similarity_scores.append(similarity_score)
        distance_scores = euclidean_distances([vector1], [vector2])[0][0]
        euclidean_distance_scores.append(distance_scores)

    euclidean_distance_scores = list(map(lambda x: 1-x, euclidean_distance_scores))

    print("Cosine_similarity")
    print(cosine_similarity_scores)

    print("Eucledian Distances")
    print(euclidean_distance_scores)
    print("")

    # 3. B)
    # Calculate the coefficient of correlation between the values of the given dataset mean
    # and the calculated values from cosine_similarity and euclidean_distance

    # correlation between calculated coefficients
    pearson_corr, _ = pearsonr(cosine_similarity_scores, euclidean_distance_scores)
    print(f"Pearson Correlation Coefficient (Euclidean + Cosine scores): {pearson_corr:.2f}")
    pearson_corr, _ = pearsonr(means, euclidean_distance_scores)
    print(f"Pearson Correlation Coefficient (Euclidean + Human scores): {pearson_corr:.2f}")
    pearson_corr, _ = pearsonr(cosine_similarity_scores, means)
    print(f"Pearson Correlation Coefficient (Cosine + Human scores): {pearson_corr:.2f}")

    print("")

    kendall_corr, _ = kendalltau(cosine_similarity_scores, euclidean_distance_scores)
    print(f"Kendall Tau Correlation Coefficient (Euclidean + Cosine scores): {kendall_corr:.2f}")
    kendall_corr, _ = kendalltau(means, euclidean_distance_scores)
    print(f"Kendall Tau Correlation Coefficient (Euclidean + Human scores): {kendall_corr:.2f}")
    kendall_corr, _ = kendalltau(cosine_similarity_scores, means)
    print(f"Kendall Tau Correlation Coefficient (Cosine + Human scores): {kendall_corr:.2f}")
