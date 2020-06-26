import autocomplete
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from autocomplete import models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import re
from spellchecker import SpellChecker
import time
import math
import pprint

_start = time.time()

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


def __extract_node_tokens(node, remove_stopwords=True):
    title = str(node['name']).lower()
    description = str(node['description']).lower()
    term_names = map(lambda x: x['name'], node['subjects'])

    # replace dashed names with a space
    title.replace('-', ' ')
    description.replace('-', ' ')
    term_names = list(map(lambda x: x.replace('-', ' '), term_names))

    term_tokens = []
    for name in term_names:
        term_tokens.extend(__preprocess_text(name, remove_stopwords))

    tokens = term_tokens + __preprocess_text(
        title, remove_stopwords
    ) + __preprocess_text(
        title, remove_stopwords
    ) + __preprocess_text(
        description, remove_stopwords
    )

    # remove tokens that aren't in wordnet
    # for i, token in enumerate(tokens):
    #     if token not in wn.all_lemma_names():
    #         print('Excluding non word ' + tokens[i])
    #         del tokens[i]

    return tokens


def __preprocess_text(text, remove_stopwords=False):
    # remove dashes
    text = text.replace('-', ' ')
    tokens = word_tokenize(str(text).lower())

    # filter out things that aren't pure letters
    letters_pattern = re.compile(r"([a-z]|[A-Z])+")
    tokens = [t for t in tokens if letters_pattern.match(t)]

    # filter out stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens


def __obtain_training_models(nodes):
    tokens = []

    for node in nodes:
        node_tokens = __extract_node_tokens(node, remove_stopwords=False)
        lemmas = list(map(lambda x: lemmatizer.lemmatize(x), node_tokens))

        tokens.extend(lemmas)

        # store lemmas
        node_id = node['_id']
        map_document_to_term_count[node_id] = {}
        map_document_to_term_list[node_id] = []
        map_document_to_total_term_count[node_id] = len(lemmas)

        # store number of documents for a given lemma
        for lemma in lemmas:
            if lemma not in map_term_to_document_count.keys():
                map_term_to_document_count[lemma] = list()

            map_term_to_document_count[lemma].append(node_id)
            map_document_to_term_count[node_id][lemma] = map_document_to_term_count[node_id].get(lemma, 0) + 1

            map_document_to_term_list[node_id].append(lemma)

    return tokens


def __preprocess_search(query):
    # create tokens
    search_tokens = __preprocess_text(query, remove_stopwords=False)

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


def __gather_relevant_documents(search_lemmas):
    all_documents = set()

    for lemma in search_lemmas:
        documents_containing_lemma = map_term_to_document_count.get(lemma, list())
        all_documents.update(set(documents_containing_lemma))

    print(all_documents)

    return all_documents


def __do_search(search_lemmas):
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

            # NOW! WEIGHT HEAVIER DOCUMENTS WHOSE OCCURANCES
            # OF THE MULTIPLE TERMS ARE CLOSER TOGETHER.
            all_doc_terms = map_document_to_term_list[doc_id]
            map_term_to_indices = {}

            # store a map of search lemma to indexes of their occurrences
            for lemma in search_lemmas:
                occurrences = [i for i, t in enumerate(all_doc_terms) if t == lemma]
                if occurrences:
                    map_term_to_indices[lemma] = occurrences

            # attempt to find a path of the smallest 'width' that hits one index for each of the terms.
            # it's easier to do this if we sort the term to indices with greatest # of indices first,
            # as that number will be the number of paths we attempt to create.
            sorted_indices = sorted(map_term_to_indices.values(), key=lambda item: len(item))
            sorted_indices.reverse()

            print(sorted_indices)
            path_scores = []

            # create a path for each of the starting indexes.
            for starting_index in sorted_indices[0]:
                # the selected indices for this path
                path_indexes = [starting_index]

                # for each subsequent layer, find the narrowest path from the previous layers
                # then add that index to path_indexes
                for i in range(1, len(sorted_indices)):
                    layer_indices = sorted_indices[i]

                    best_index = -1
                    smallest_width = math.inf
                    for potential_index in layer_indices:
                        potential_indexes = path_indexes + [potential_index]
                        potential_width = max(potential_indexes) - min(potential_indexes)

                        if best_index == -1 or potential_width < smallest_width:
                            best_index = potential_index
                            smallest_width = potential_width

                    path_indexes.append(best_index)

                path_width = max(path_indexes) - min(path_indexes)
                path_score = math.log(len(all_doc_terms) / path_width)
                print("-->".join(map(lambda x: str(x), path_indexes)) + ': ' + str(path_width) + ' @' + str(path_score))
                path_scores.append(path_score)

            doc_proximity_mult = (sum(path_scores) / len(path_scores))
            print('Proximity Mult ' + str(doc_proximity_mult))
            document_scores[doc_id] *= doc_proximity_mult

    # weight heavier documents that

    threshold = 0.1
    filtered_results = filter(lambda x: x[1] >= threshold, document_scores.items())
    filtered_results = sorted(filtered_results, key=lambda item: item[1])
    filtered_results.reverse()

    # determine whether there are 'too many' to count
    # i.e. all have been filtered out due to low score and there were originally items
    if not filtered_results and document_scores.items():
        print('Try a more specific query.')

    result_tuples = []

    for result in filtered_results:
        res_id = result[0]
        res_score = result[1]
        res_name = map_node_id_to_node[res_id]["name"]
        res_desc = map_node_id_to_node[res_id]["description"]
        result_tuples.append((res_score, res_name, res_desc))

    return result_tuples


map_term_to_document_count = {}
map_document_to_term_count = {}
map_document_to_term_list = {}
map_document_to_total_term_count = {}

# Train models
training_tokens = __obtain_training_models(course_nodes)
models.train_models(" ".join(training_tokens))
spell.word_frequency.load_words(training_tokens)

# print(map_term_to_document_count)
# print(map_document_to_term_count)

print('Loaded models in ' + str((time.time() - _start) / 1000) + 'ms')

# pprint.pprint(map_term_to_document_count)
# pprint.pprint(map_document_to_term_count)

_start = time.time()


def search(query):
    search_lemmas = __preprocess_search(query)
    print(search_lemmas)
    return __do_search(search_lemmas)

# print(len(map_term_to_document_count['military']))
# print(len(map_term_to_document_count['chinese']))
