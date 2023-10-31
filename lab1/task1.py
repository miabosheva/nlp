import nltk
import string
import re
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
file_path = "C:/Users/Mia/Desktop/FINKI/NLP/lab1/train_en.txt"

def format_text(text):
    sentences = re.split(r'\d+', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences

if __name__ == "__main__":
    with open(file_path, 'r') as file:
        text = file.read()
    # razdeli recenici na sekoja 0 ili 1
    sentences = format_text(text)

    # 1. A)
    words = []
    # razdeluvanje na zborovi
    for sentence in sentences:
        temp = word_tokenize(sentence)
        words += temp

    # print(words)
    words_clean = [word.lower() for word in words if word.isalnum()]
    word_frequencies = FreqDist(words_clean)
    vocab = word_frequencies.keys()

    # print all fords from vocabulary
    # print(list(vocab))

    # for word, count in word_frequencies.most_common(10):
    #     print(f"{word}: {count}")

    old_vocab_size = len(vocab)
    print("Old vocab size: " + str(old_vocab_size))

    stop_words = set(stopwords.words('english'))

    # 1. B)
    # Remove stopwords and punctuation from the list of words
    words_new = [word for word in words_clean
                      if word.lower() not in stop_words
                      and word not in string.punctuation]
    word_frequencies_clean = FreqDist(words_new)

    for word, count in word_frequencies_clean.most_common(10):
        print(f"{word}: {count}")

    vocab = word_frequencies_clean.keys()
    new_vocab_size = len(vocab)
    print("New vocab size: " + str(new_vocab_size))





