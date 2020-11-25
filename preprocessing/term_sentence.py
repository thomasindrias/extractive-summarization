import numpy as np


def normalized_terms(tokens):
    """
    Returns a list of all alphabetic (non-numeric, non-punctuation) terms,
    disregarding stop words as defined by the Spacy library. The individual terms
    will be lemmatized, i.e. in base dictionary form.

    ## Parameters:
    tokens: A collection of Spacy tokens e.g. from a document or sentence

    ## Returns:
    normalized_terms: A list of lemmatized, non-stop-word terms in `tokens`,
    preserving their original order 
    """
    return [token.lemma_ for token in tokens if (not token.is_stop) and token.is_alpha]


def normalized_sentences(doc):
    """
    Returns a list of the sentences in `doc` with normalized terms, as specified by the `normalized_terms` function.

    ## Parameters:
    doc: A document which has been processed by some NLP model, e.g. Spacy

    ## Returns:
    normalized_sentences: A list of the sentences given in `doc`, each consisting only of the relevant terms determined by the `normalized_terms` function
    """
    return [normalized_terms(sentence) for sentence in doc.sents]


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

    for j, sentence in enumerate(sentences):
        for term in sentence:
            # Find the term indices to increment the proper element in sentence column
            i = np.where(terms == term)
            A[i, j] += 1

    return A
