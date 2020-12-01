import numpy as np
from preprocessing.term_sentence import normalized_terms
from preprocessing.term_sentence import normalized_sentences


class Document:
    raw_content = ''
    nlp_content = None
    terms = []
    sentences = []
    raw_sentences = []

    def __init__(self, filename, nlp):
        """
        Reads text content from the given filename then processes this with the given NLP object.
        The `terms` and `sentences` obtained through the NLP object are made available as member fields,
        and the general result of the NLP process is available via the `nlp_content` field.

        ## Parameters
        filename: A path to the text document to read from.
        nlp: A natural language processing model, e.g. using the Spacy library.
        """
        with open(filename) as f:
            self.raw_content = f.read()

        # Process the document using the given nlp-model
        self.nlp_content = nlp(self.get_normalized_content())

        # Extract normalized terms and as sentences from the document
        self.terms = np.unique(normalized_terms(self.nlp_content))
        self.sentences = normalized_sentences(self.nlp_content)
        self.raw_sentences = list(self.nlp_content.sents)

    def get_normalized_content(self):
        """
        Performs any necessary preprocessing on the raw document content, given in `raw_content`, before NLP.
        """
        return self.raw_content.lower()
