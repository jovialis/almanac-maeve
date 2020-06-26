import autocomplete
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from autocomplete import models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker
import time
import math
import pprint

start = time.time()

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker(distance=1)  # set at initialization

client = MongoClient()
modeling_db = client.modellingDB
course_nodes = list(modeling_db.nodes.find({
    'nodeType': 'CourseNode'
}))

map_node_id_to_node = {}
for node in course_nodes:
    map_node_id_to_node[node['_id']] = node


def extract_node_tokens(node, remove_stopwords=True):
    title = str(node['name']).lower()
    description = str(node['description']).lower()
    term_names = map(lambda x: x['name'], node['subjects'])

    # replace dashed names with a space
    title.replace('-', ' ')
    description.replace('-', ' ')
    term_names = list(map(lambda x: x.replace('-', ' '), term_names))

    term_tokens = []
    for name in term_names:
        term_tokens.extend(preprocess_text(name, remove_stopwords))

    return term_tokens + preprocess_text(title, remove_stopwords) + preprocess_text(title, remove_stopwords) + preprocess_text(description, remove_stopwords)


def preprocess_text(text, remove_stopwords=False):
    tokens = word_tokenize(str(text).lower())

    # filter out things that aren't pure letters
    letters_pattern = re.compile(r"([a-z]|[A-Z])+")
    tokens = [t for t in tokens if letters_pattern.match(t)]

    # filter out stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens


map_term_to_document_count = {}
map_document_to_term_count = {}
map_document_to_total_term_count = {}


def obtain_training_models(nodes):
    tokens = []

    for node in nodes:
        node_tokens = extract_node_tokens(node, remove_stopwords=False)
        lemmas = list(map(lambda x: lemmatizer.lemmatize(x), node_tokens))

        tokens.extend(lemmas)

        # store lemmas
        node_id = node['_id']
        map_document_to_term_count[node_id] = {}
        map_document_to_total_term_count[node_id] = len(lemmas)

        # store number of documents for a given lemma
        for lemma in lemmas:
            if lemma not in map_term_to_document_count.keys():
                map_term_to_document_count[lemma] = list()

            map_term_to_document_count[lemma].append(node_id)
            map_document_to_term_count[node_id][lemma] = map_document_to_term_count[node_id].get(lemma, 0) + 1

    return tokens


# Train models
training_tokens = obtain_training_models(course_nodes)
models.train_models(" ".join(training_tokens))
spell.word_frequency.load_words(training_tokens)

# print(map_term_to_document_count)
# print(map_document_to_term_count)

print('Loaded models in ' + str((time.time() - start) / 1000) + 'ms')


def preprocess_search(query):
    # create tokens
    search_tokens = preprocess_text(query, remove_stopwords=False)

    for i, token in enumerate(search_tokens):
        # attempt to spellcheck invalid tokens
        if len(token) >= 3:
            if spell.unknown([token]):
                search_tokens[i] = spell.correction(token)
                # print('corrected ' + token + ' to ' + search_tokens[i])
                token = search_tokens[i]

        if i > 0:
            prev = search_tokens[i - 1]
            pred = autocomplete.predict(prev, token)
        else:
            pred = autocomplete.predict_currword(token)

        if len(pred) > 0:
            best_pred = pred[0][0]
        else:
            best_pred = token

        search_tokens[i] = lemmatizer.lemmatize(best_pred)

    # remove stopwords if we have non-stopword terms. otherwise search purely by stopwords as a last case.
    non_stopwords = [x for x in search_tokens if x not in stopwords]
    if non_stopwords:
        search_tokens = non_stopwords

    return search_tokens


def gather_relevant_documents(search_lemmas):
    all_documents = set()

    for lemma in search_lemmas:
        documents_containing_lemma = map_term_to_document_count.get(lemma, list())
        all_documents.update(set(documents_containing_lemma))

    print(all_documents)

    return all_documents


def do_search(search_lemmas):
    document_scores = {}

    # corpus of potential documents for this search.
    # narrow the relevant documents to the entire list if we have multiple keyword lemmas
    # if len(search_lemmas) > 1:
    #     relevant_document_count = len(gather_relevant_documents(search_lemmas))
    # else:
    #     relevant_document_count = len(course_nodes)

    relevant_document_count = len(course_nodes)

    # calculate tf-idf
    for lemma in search_lemmas:
        documents_containing_lemma = map_term_to_document_count.get(lemma, list())

        if not documents_containing_lemma:
            continue

        idf = math.log(relevant_document_count / len(documents_containing_lemma))
        # print(lemma + ' idf : ' + str(idf))

        # calculate idf for each document
        for doc_id in documents_containing_lemma:
            # (Number of times term t appears in a document) / (Total number of terms in the document).
            lemma_count_in_doc = map_document_to_term_count[doc_id][lemma]
            total_count_in_doc = map_document_to_total_term_count[doc_id]

            # print(total_count_in_doc)

            tf = lemma_count_in_doc / total_count_in_doc
            document_scores[doc_id] = document_scores.get(doc_id, 0) + (tf * idf)

            # print(str(doc_id) + ' has a score of ' + str(tf*idf) + ' for ' + lemma)

    # weight heavier documents that have multiple of the desired terms.
    for doc_id in document_scores.keys():
        overlap = [k for k in map_document_to_term_count[doc_id].keys() if k in search_lemmas]

        # print(overlap)
        # print(document_scores[doc_id])

        # if more than one of the search lemma are present, we weight the node heavier.
        if len(overlap) > 1:
            weight = math.exp((len(overlap) / len(search_lemmas)) + 2)
            document_scores[doc_id] *= weight

        # print(document_scores[doc_id])

    threshold = 0.1
    filtered_results = filter(lambda x: x[1] >= threshold, document_scores.items())
    filtered_results = sorted(filtered_results, key=lambda item: item[1])
    filtered_results.reverse()

    # determine whether there are 'too many' to count
    # i.e. all have been filtered out due to low score and there were originally items
    if not filtered_results and document_scores.items():
        print('Try a more specific query.')

    for result in filtered_results:
        res_id = result[0]

        print('----------')
        print('score ' + str(result[1]))
        print('name ' + str(map_node_id_to_node[res_id]["name"]))
        print('desc ' + str(map_node_id_to_node[res_id]["description"]))


# pprint.pprint(map_term_to_document_count)
# pprint.pprint(map_document_to_term_count)

start = time.time()

SEARCH_QUERY = "econ"
search_lemmas = preprocess_search(SEARCH_QUERY)
print(search_lemmas)
do_search(search_lemmas)

print('Searched in ' + str((time.time() - start)) + 's')

# print(len(map_term_to_document_count['military']))
# print(len(map_term_to_document_count['chinese']))
