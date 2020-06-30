import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class AnalyzeContent:

    # accepts node_contents as an array of tuples in the form
    # ('text', 1) where text is a chunk of text and 1 is the text weight
    # relative to the entire node
    def __init__(self, text_weights):
        self.contents = text_weights

    # preprocesses a chunk of text, removing symbols and
    # separating dashed words.
    @staticmethod
    def __preprocess_tokenize_text(text, remove_stopwords=False):
        # lowercase
        text = text.lower()

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

    # lemmatizes a list of tokens
    @staticmethod
    def __lemmatize_tokens(tokens):
        lemmas = list(map(lambda x: lemmatizer.lemmatize(x), tokens))
        return lemmas

    # returns a flat array of lemmatized tokens in the order they were provided.
    # does NOT flatten based on weight, purely on contents
    def content_tokens_flattened_raw(self):
        # create a single string from all provided text
        all_text = list(map(lambda x: x[0], self.contents))
        all_text = " ".join(all_text)

        # tokenize then lemmatize
        tokens = self.__preprocess_tokenize_text(all_text, False)
        lemmas = self.__lemmatize_tokens(tokens)

        return lemmas

    # returns a dictionary mapping term to their relative weight in the document,
    def content_token_weight_map(self):
        # count and store frequency
        frequency_map = {}

        # print(self.contents)

        # iterate through and weight all tokens accordingly
        for content in self.contents:
            text = content[0]
            weight = content[1]

            # tokenize the text
            text_tokens = self.__preprocess_tokenize_text(text)
            text_tokens = self.__lemmatize_tokens(text_tokens)

            for token in text_tokens:
                frequency_map[token] = frequency_map.get(token, 0) + (1 * weight)

        return frequency_map

    # returns a list of unique terms from this document
    def content_token_list(self):
        frequency = self.content_token_weight_map()
        return list(frequency.keys())

    # returns a count of unique terms
    def content_token_count(self):
        return len(self.content_token_list())
