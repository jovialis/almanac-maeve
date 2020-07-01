from datetime import datetime
import math
import os

from pymongo import MongoClient
from autocomplete import models
from spellchecker import SpellChecker
import autocomplete
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from analyzecontent import AnalyzeContent


class Searcher:

    def __init__(self):
        # status variables
        self._corpus_size = 0
        self._trained = False

        # utils
        self._spellcheck = SpellChecker(distance=1)  # set at initialization
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words('english'))

        # instantiate db
        _client = MongoClient(os.environ.get('MONGODB_URI'))
        _search_db = _client[os.environ.get('SEARCH_DB')]
        _modeling_db = _client[os.environ.get('NODE_DB')]

        self._indexes_collection = _search_db['docs']
        self._lemmas_collection = _search_db['lemmas']
        self._cache_collection = _search_db['cached']
        self._nodes_collection = _modeling_db['nodes']

    def train(self):
        print('Training searcher...')
        start = datetime.now()

        tokens_corpus = []

        # create corpus from indexed docs
        doc_indexes = list(self._indexes_collection.find({}))

        # extend corpus with node body tokens
        for node in doc_indexes:
            tokens_corpus.extend(node["body"])

        # extend corpus with 10,000 of the most commonly used english words
        text_file = open('resources/english-words.txt')
        english_words = text_file.read().split('\n')
        tokens_corpus.extend(english_words)
        text_file.close()

        # perform training of our models
        models.train_models(" ".join(tokens_corpus))
        self._spellcheck.word_frequency.load_words(tokens_corpus)

        self._trained = True
        self._corpus_size = len(doc_indexes)
        print('Finished training in ' + (str(datetime.now() - start)) + 's')

    # performs a search with a strong query
    def search(self, query):
        if not self._trained:
            print("Search must be trained first upon instantiation.")
            return

        query_tokens = self.__preprocess_search_query(query)

        # ignore empty queries
        if not query_tokens:
            return []

        # discover cached query
        cached_query = self.__load_cached_query(query_tokens)
        if cached_query is not None:
            return cached_query

        # otherwise, return our nodes and cache
        node_ids = self.__perform_search(query_tokens)
        nodes = self.__lookup_nodes(node_ids)

        # cache results
        self.__save_query_to_cache(query_tokens, nodes)

        return nodes

    def __preprocess_search_query(self, query):
        # create tokens
        consumed = AnalyzeContent(text_weights=[(query, 1)])
        search_tokens = consumed.content_token_list()

        for i, token in enumerate(search_tokens):
            # attempt to spellcheck invalid tokens
            if len(token) >= 3:
                if self._spellcheck.unknown([token]):
                    search_tokens[i] = self._spellcheck.correction(token)
                    # print('corrected ' + token + ' to ' + search_tokens[i])
                    token = search_tokens[i]

            if i > 0 and autocomplete.predict_currword(search_tokens[i - 1]):
                prev = search_tokens[i - 1]
                pred = autocomplete.predict(prev, token)
            else:
                pred = autocomplete.predict_currword(token)

            if len(pred) > 0:
                best_pred = pred[0][0]
            else:
                best_pred = token

            search_tokens[i] = self._lemmatizer.lemmatize(best_pred)

        # remove stopwords if we have non-stopword terms. otherwise search purely by stopwords as a last case.
        non_stopwords = [x for x in search_tokens if x not in self._stopwords]
        if non_stopwords:
            search_tokens = non_stopwords

        return search_tokens

    # attempts to discover a cached query
    def __load_cached_query(self, search_lemmas):
        # find cached result by terms
        search_lemmas = sorted(search_lemmas)
        cached_query = self._cache_collection.find_one({"terms": search_lemmas})
        if not cached_query:
            return None

        return cached_query["results"]

    # caches a search query
    def __save_query_to_cache(self, search_lemmas, results):
        # saves query to cache
        search_lemmas = sorted(search_lemmas)
        self._cache_collection.insert_one({
            "terms": search_lemmas,
            "results": results
        })

    def __perform_search(self, search_lemmas):
        # print(search_lemmas)
        # database documents necessary for search

        # discover all documents with at least one of the search lemmas.
        # we perform this query once instead of for each lemma to cut down on runtime
        lemma_query = {"$or": list(map(lambda x: {"lemma": x}, search_lemmas))}

        # query to lookup all lemma docs and populate their documents
        all_lemma_docs = self._lemmas_collection.aggregate([{
            "$match": lemma_query
        }, {
            "$lookup": {
                "from": "docs",
                "localField": "docs",
                "foreignField": "_id",
                "as": "docs"
            }
        }])

        doc_scores = {}
        doc_id_to_contents = {}

        # iterate over the lemma docs, calculate tf-idf for all referenced docs
        for lemma_doc in all_lemma_docs:
            lemma = lemma_doc["lemma"]

            # extract the docs containing the lemma
            docs_containing_lemma = lemma_doc["docs"]
            if not docs_containing_lemma:
                continue

            idf = math.log(self._corpus_size / len(docs_containing_lemma))

            # iterate over all the docs containing the lemma
            for doc in docs_containing_lemma:
                # pull doc id
                doc_id = doc["_id"]

                # store doc for later reference
                if doc_id not in doc_id_to_contents.keys():
                    doc_id_to_contents[doc_id] = doc

                # pull weights and calculate tf
                lemma_weight = float(doc["lemmaWeights"][lemma])
                body_length = float(doc["bodyLength"])

                tf = lemma_weight / body_length
                doc_scores[doc_id] = doc_scores.get(doc_id, float(0)) + (tf * idf)

        # weight heavier documents that have multiple of the desired terms.
        for doc_id in doc_scores.keys():
            doc = doc_id_to_contents[doc_id]

            # determine how MANY of the the desired lemmas are in this doc
            overlap = [k for k in doc["lemmas"] if k in search_lemmas]

            # print(overlap)
            # print(document_scores[doc_id])

            # if more than one of the search lemma are present, we weight the node heavier.
            if len(overlap) > 1:
                weight = math.exp((len(overlap) / len(search_lemmas)) + 2)
                doc_scores[doc_id] *= weight

                # NOW! WEIGHT HEAVIER DOCUMENTS WHOSE OCCURANCES
                # OF THE MULTIPLE TERMS ARE CLOSER TOGETHER.
                doc_body = doc["body"]
                map_lemma_to_indices = {}

                # store a map of search lemma to indexes of their occurrences
                for lemma in search_lemmas:
                    occurrences = [i for i, t in enumerate(doc_body) if t == lemma]
                    if occurrences:
                        map_lemma_to_indices[lemma] = occurrences

                # attempt to find a path of the smallest 'width' that hits one index for each of the terms.
                # it's easier to do this if we sort the term to indices with greatest # of indices first,
                # as that number will be the number of paths we attempt to create.
                sorted_indices = sorted(map_lemma_to_indices.values(), key=lambda item: len(item))
                sorted_indices.reverse()

                # print(sorted_indices)
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
                    path_score = math.log(len(doc_body) / path_width)
                    # print("-->".join(map(lambda x: str(x), path_indexes)) + ': ' + str(path_width) + ' x' + str(path_score))
                    path_scores.append(path_score)

                doc_proximity_mult = (sum(path_scores) / len(path_scores))
                # print('Proximity Mult x' + str(doc_proximity_mult))
                doc_scores[doc_id] *= doc_proximity_mult

        # filter out docs below threshold then sort by score
        threshold = 0.1
        filtered_results = filter(lambda x: x[1] >= threshold, doc_scores.items())
        filtered_results = sorted(filtered_results, key=lambda item: item[1])
        filtered_results.reverse()

        # determine whether there are 'too many' to count
        # i.e. all have been filtered out due to low score and there were originally items
        if not filtered_results and doc_scores.items():
            return "vague"

        # return sorted list of document ids mapped to node ids
        return list(map(lambda x: doc_id_to_contents[x[0]]["node"], filtered_results))

    def __lookup_nodes(self, node_id_list):
        # create array of referenced nodes, projecting weight by their index
        nodes = list(self._nodes_collection.aggregate([{
            # match docs
            "$match": {
                "_id": {"$in": node_id_list}
            }
        }, {
            # project weight
            "$addFields": {
                "__weight": {
                    "$indexOfArray": [
                        node_id_list,
                        "$_id"
                    ]
                }
            }
        }, {
            # sort by weight
            "$sort": {
                "__weight": 1
            }
        }]))

        return nodes
