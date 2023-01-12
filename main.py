import os
import re
from math import sqrt
import pandas as pd
import ast
import MeCab
from gensim import corpora
from gensim import matutils
from gensim import models
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


dirname_temp = os.path.dirname(__file__)
templates = '/sample/'
dirname = dirname_temp + templates
# This CSV has already processed Morphological analysis(['Name',...],[Classification]).
data_csv = 'sample.csv'
id2word_txt = 'sample1.txt'  # ID-Word-Frequent
basename_stopwords = 'sample2.txt'  # Stop word
basename_mecab = 'sample.dic'  # User dictionary of Mecab
filename_data_csv = os.path.join(dirname, data_csv)
filename_id2word_txt = os.path.join(dirname, id2word_txt)
filename_mecab = os.path.join(dirname, basename_mecab)


input_data = []  # Please input sentece contains adverse events or not.


# Convert from sparse gensim format to dense list of numbers
def vec2dense(vec, num_terms) -> list:
    return list(matutils.corpus2dense([vec], num_terms=num_terms).T[0])


df = pd.read_csv(filename_data_csv)

words_list = [ast.literal_eval(d) for d in df['names']]

result_list = []
for results in df['classified']:
    result_list.append(results)

docs = {}
for key, value in zip(result_list, words_list):
    docs[key] = value
names = docs.keys()


# Create a dictionary
dct = corpora.Dictionary(docs.values())
dct.filter_extremes(no_below=20, no_above=0.05)
dct.save_as_text(filename_id2word_txt)

# BoW(Bag of Words)
bow_docs = {}
bow_docs_all_zeros = {}
for name in names:
    sparse = dct.doc2bow(docs[name])
    bow_docs[name] = sparse
    dense = vec2dense(sparse, num_terms=len(dct))

# LSI model
lsi_docs = {}
num_topics: int = 2
lsi_model = models.LsiModel(bow_docs.values(), id2word=dct.load_from_text(
    filename_id2word_txt), num_topics=num_topics)

for name in names:
    vec = bow_docs[name]
    sparse = lsi_model[vec]
    dense = vec2dense(sparse, num_topics)
    lsi_docs[name] = sparse

# Normalize a vector
unit_vecs = {}
for name in names:
    vec = vec2dense(lsi_docs[name], num_topics)
    norm = sqrt(sum(num**2 for num in vec))
    unit_vec = [num / norm for num in vec]
    unit_vecs[name] = unit_vec


# Binary classification
all_data = [unit_vecs[name] for name in names if re.match('NonAE', name)]
all_data.extend([unit_vecs[name]
                for name in names if re.match('AE', name)])

all_labels = [0 for name in names if re.match('NonAE', name)]
all_labels.extend([1 for name in names if re.match('AE', name)])

X_train, X_test, y_train, y_test = train_test_split(
    all_data, all_labels)

# Train SVM classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Prediction
predict_label = classifier.predict(X_test)
target_names = ["class 'NonAE'", "class 'AE'"]
classification_report(y_test, predict_label, target_names=target_names)


# Confirmation of unknown data
def make_flame() -> list:

    keywords = []
    for row in input_data:
        keyword = ''.join(row)
        keywords.append(keyword)

    df = pd.DataFrame(keywords)
    df = df.rename(columns={0: 'texts'})

    return df


make_flames = make_flame()


def mecab_text(text) -> list:

    mecab = MeCab.Tagger(
        r'-d /mecab/-u {}'.format(filename_mecab))
    mecab.parse('')
    node = mecab.parseToNode(text)

    wordlist = []
    while node:
        if node.feature.split(',')[0] == '名詞':
            wordlist.append(node.surface)
        node = node.next
    return wordlist


make_flames['words'] = make_flames['texts'].apply(mecab_text)


def inputdata_decision() -> str:

    pre: list = ['NOT contains', 'contains']

    for wakati_input_texts in make_flames['words']:
        dct = corpora.Dictionary([wakati_input_texts])
        tuple_sparse = dct.doc2bow(wakati_input_texts)
        sparse_lsi: list = lsi_model[tuple_sparse]
        list_dense: list = vec2dense(sparse_lsi, num_topics)
        list_norm = sqrt(sum(num**2 for num in list_dense))
        unit_vec_unlisted: list = [num / list_norm for num in list_dense]

        # Result
        Prediction_result: list = classifier.predict([unit_vec_unlisted])

    return print(f'Result:This description {pre[int(Prediction_result)]} adverse events.')
