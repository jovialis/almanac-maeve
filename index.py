import pprint
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from spellchecker import SpellChecker

from pymongo import MongoClient

spellcheck = SpellChecker()

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language="english")

client = MongoClient()

modeling_db = client.modellingDB
course_nodes = modeling_db.nodes.find({
    'nodeType': 'CourseNode'
})

first_node = course_nodes[0]

# Search algorithm:
#  1) get input text
#  2) tokenize input text
#  3) spellcheck all tokens
#  4)
#
#  3) EXACT: Search for
#  1) look for exact


def parse_node(node):
    tokens = extract_node_tokens(node)

    # lemmatize tokens
    lemmas = list(map(lambda x: lemmatizer.lemmatize(x), tokens))

    # stem tokens
    stems = list(map(lambda x: stemmer.stem(x), lemmas))

    print('Lemmas: ' + str(lemmas))
    print('Stems: ' + str(stems))


def extract_node_tokens(node):
    title = node['name'].lower()
    description = node['description'].lower()

    tokens = word_tokenize(title) + word_tokenize(description)

    # filter out things that aren't pure letters
    letters_pattern = re.compile(r"([a-z]|[A-Z])+")
    tokens = [t for t in tokens if letters_pattern.match(t)]

    # filter out stopwords
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


parse_node(first_node)
