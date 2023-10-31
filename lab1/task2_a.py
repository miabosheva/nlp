import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

file_path = "C:/Users/Mia/Desktop/FINKI/NLP/lab1/train_en.txt"

def format_text():
    with open(file_path, 'r') as file:
        text = file.read()
    # razdeli recenici na sekoja 0 ili 1
    sentences = re.split(r'\d+', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences

def get_model():
    sentences = format_text()
    df = pd.DataFrame(sentences, columns=['sentences'])
    df['sentences'] = df['sentences'].apply(lambda x: word_tokenize(x.lower()))

    # print(df['sentences'].iloc[0:10])
    sentences = df['sentences'].values

    model_word2vec = Word2Vec(sentences, vector_size=50, window=3, sg=1, min_count=15)
    # model_word2vec.save("model_word2vec.bin")
    return model_word2vec

if __name__ == "__main__":
    model = get_model()
