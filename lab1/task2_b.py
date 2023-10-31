from gensim.models import KeyedVectors
import gensim.downloader as downloader

if __name__ == "__main__":
    glove_file_path = 'glove.6B.300d.w2vformat.txt'

    # Load the GloVe embedding file
    word_vectors = KeyedVectors.load_word2vec_format(glove_file_path, binary=False)

    # check if all words are in gloves model vocabulary
    words_to_check = ['bigger', 'colder', 'windows', 'microsoft', 'king', 'queen','google']
    for word in words_to_check:
        if word not in word_vectors:
            print(f"'{word}' not found in the vocabulary.")
            exit()

    # Paris - France + Italy
    result_vector = word_vectors['paris'] - word_vectors['france'] + word_vectors['italy']
    similar_word, similarity_score = word_vectors.similar_by_vector(result_vector, topn=1)[0]
    print("Word closest to 'Paris - France + Italy':", similar_word)
    #print("Similarity Score:", similarity_score)

    # Madrid - Spain + France
    result_vector = word_vectors['madrid'] - word_vectors['spain'] + word_vectors['france']
    similar_word, similarity_score = word_vectors.similar_by_vector(result_vector, topn=1)[0]
    print("Word closest to 'Madrid - Spain + France':", similar_word)
    # print("Similarity Score:", similarity_score)

    # King - Man + Woman
    result_vector = word_vectors['king'] - word_vectors['man'] + word_vectors['woman']
    similar_word, similarity_score = word_vectors.similar_by_vector(result_vector, topn=1)[0]
    print("Word closest to 'King - Man + Woman':", similar_word)
    # print("Similarity Score:", similarity_score)

    # Bigger - Big + Cold
    result_vector = word_vectors['bigger'] - word_vectors['big'] + word_vectors['cold']
    similar_word, similarity_score = word_vectors.similar_by_vector(result_vector, topn=1)[0]
    print("Word closest to 'Bigger - Big + Cold':", similar_word)
    # print("Similarity Score:", similarity_score)

    # Windows - Microsoft + Google
    result_vector = word_vectors['windows'] - word_vectors['microsoft'] + word_vectors['google']
    similar_word, similarity_score = word_vectors.similar_by_vector(result_vector, topn=1)[0]
    print("Word closest to 'Windows - Microsoft + Google':", similar_word)
    # print("Similarity Score:", similarity_score)
