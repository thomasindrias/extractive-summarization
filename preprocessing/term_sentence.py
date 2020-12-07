import numpy as np


def is_valid_length(token):
    """
    Defines whether or not a given token is of appropriate length.

    ## Returns:
    `True` if the token is longer then two characters but shorter than fifteen, `False` otherwise
    """
    return len(token) > 2 and len(token) < 15


def normalized_terms(tokens, lemmatizer, stop_words):
    """
    Returns a list of all alphabetic (non-numeric, non-punctuation) terms,
    disregarding the given stop words. The individual terms
    will be lemmatized, i.e. in base dictionary form according to the given lemmatizer.

    ## Parameters:
    tokens: A collection of Spacy tokens e.g. from a document or sentence
    lemmatizer: A natural language processing lemmatizer, e.g. using the WordNetLemmatizer from NLTK. Must have a method `lemmatize`.
    stop_words: A list of stop words to ignore when extracting terms.

    ## Returns:
    normalized_terms: A list of lemmatized, non-stop-word terms in `tokens`,
    preserving their original order 
    """
    return np.array([
        lemmatizer.lemmatize(token.lower()) for token in tokens
        if token not in stop_words and token.isalpha() and is_valid_length(token)
    ])


def create_term_sentence(terms, sentences):
    """
    Constructs a term-sentence matrix where each column corresponds to one sentence and each row corresponds to a term.
    The element at row *i* and column *j* will correspond to the frequency of term *i* in sentence *j*.

    ## Parameters:
    terms: A list of terms from some NLP model
    sentences: A list of sentences consisting of the terms given in `terms`

    ## Returns:
    A: The term-sentence matrix
    """
    # Construct the term-sentence matrix A
    # This will be a *highly* sparse matrix, meaning lots of wasted memory :c
    A = np.zeros((len(terms), len(sentences)))
    df = np.zeros(len(terms))

    for j, sentence in enumerate(sentences):
        for i, term in enumerate(terms):
            # Compute document frequency for each term
            if term in sentence:
                df[i] += 1

        for term in sentence:
            # Find the term indices to increment the proper element in sentence column
            i = np.where(terms == term)
            A[i, j] += 1

        # Divide frequency by word occurence in the sentence
        # A[:, j] /= A[:, j].size

    # Compute the IDF weights
    idf = np.log(len(sentences)/df)
    A *= idf[:, np.newaxis]

    return A
