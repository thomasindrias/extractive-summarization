import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from preprocessing.document import Document
from preprocessing.term_sentence import create_term_sentence
from saliency.saliency_score import saliency_score
from key_sentence.key_sentence import key_sentence
import matplotlib.pyplot as plt

# English language stop words and lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')


# Read and parse a text document, in this case an article about the Covid-19 vaccine

def main(path):
    """
    docstring
    """
    
    doc = Document(path, lemmatizer, stop_words)

    # Create a term-sentence matrix from the document content
    A = create_term_sentence(doc.terms, doc.sentences)


    print(f'Number of terms: {len(doc.terms)}')
    print(f'Number of sentences: {len(doc.sentences)}')
    # print(f'Maximum term frequency in one sentence: {np.max(A)}')

    #print("\nTerms:")
    #print(doc.terms)

    #print("\nFirst column of A:")
    #print(A[:, 0])

    u, key_terms, key_sencences_sorted = saliency_score(A)
    key_sencences_sorted = key_sencences_sorted.flatten()

    k=10
    key_sencences, C = key_sentence(A, k, top=5)

    print("\n========= Key terms ============\n")

    print(doc.terms[key_terms][0][:10])
    print("\n")
    
    for (i, sentence) in enumerate(key_sencences_sorted[:5]):
        print(i, ":", doc.raw_sentences[sentence])

    print("\n========= Key sentences ============\n")

    for (i, sentence) in enumerate(key_sencences):
        print(i, ":", doc.raw_sentences[sentence])
    
    return u, C


paths = ['data/wikipedia_text_mining.txt', 'data/the_guardian_vaccine_article.txt', 'data/weirdest_computer.txt']

if __name__ == '__main__':
    for path in paths:
        print(f'Running algorithm on {path}\n')
        main(path)
