import numpy as np
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from preprocessing.term_sentence import normalized_terms

class Document:
    raw_content = ''
    terms = []
    sentences = []
    raw_sentences = []

    def __init__(self, filename, lemmatizer, stop_words):
        """
        Reads text content from the given filename then processes this using the given lemmatizer and stop word list.
        The `terms` and `sentences` obtained through this processing are made available as member fields.

        ## Parameters
        filename: A path to the text document to read from.
        lemmatizer: A natural language processing lemmatizer, e.g. using the WordNetLemmatizer from NLTK. Must have a method `lemmatize`.
        stop_words: A list of stop words to ignore when extracting terms.
        """
        with open(filename) as f:
            self.raw_content = f.read()

        # Split into sentences
        sent_tokenizer = PunktSentenceTokenizer(self.raw_content)
        self.raw_sentences = np.array(sent_tokenizer.tokenize(self.raw_content))

        # Tokenize each sentence and normalize the terms
        self.sentences = [word_tokenize(sent) for sent in self.raw_sentences]
        self.sentences = np.array([normalized_terms(
            sent, lemmatizer, stop_words) for sent in self.sentences])

        # Collect unique terms across all tokenized sentences
        self.terms = np.unique(
            [term for sent in self.sentences for term in sent])
