import numpy as np
import re
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

        self.process(lemmatizer, stop_words)


    def process(self, lemmatizer, stop_words):
        """
        Processes the raw content of the document into terms and sentences for use in forming a term-sentence matrix.
        Sets the member fields `terms`, `sentences` and `raw_sentences` based on the contents of the field `raw_content`.

        ## Parameters
        lemmatizer: A natural language processing lemmatizer, e.g. using the WordNetLemmatizer from NLTK. Must have a method `lemmatize`.
        stop_words: A list of stop words to ignore when extracting terms.
        """
        # Remove extraneous whitespace
        # self.raw_content = re.sub('[\.\W]\s\s+', '. ', self.raw_content)
        self.raw_content = re.sub('\n', ' ', self.raw_content)

        # Split into sentences
        sent_tokenizer = PunktSentenceTokenizer(self.raw_content)
        self.raw_sentences = np.array(
            sent_tokenizer.tokenize(self.raw_content))

        # Tokenize each sentence and normalize the terms
        self.sentences = [word_tokenize(sent) for sent in self.raw_sentences]
        self.sentences = np.array([normalized_terms(
            sent, lemmatizer, stop_words) for sent in self.sentences])

        # Find and discard any empty sentences at this point
        non_empty = np.where([sent.size > 0 for sent in self.sentences])
        self.raw_sentences = self.raw_sentences[non_empty]
        self.sentences = self.sentences[non_empty]

        # Collect unique terms across all tokenized sentences
        self.terms = np.unique(
            [term for sent in self.sentences for term in sent])
