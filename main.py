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

paths = ['data/wikipedia_text_mining.txt', 'data/the_guardian_vaccine_article.txt', 'data/weirdest_computer.txt']

# Read and parse a text document, in this case an article about the Covid-19 vaccine

def create_graph():
    fig, ax = plt.subplots(2, 3, figsize=(20,10), constrained_layout=True)
    fig.suptitle('Key word and key sentence scores', fontsize=16)
    
    for index, path in enumerate(paths):
        print(f'Running algorithm on {path}\n')
        u = None
        u, C = main(path)
        
        ax[0, index].plot(u)
        ax[0, index].set_xlabel('Terms', fontsize = 12)
        ax[0, index].set_ylabel('Score', fontsize = 12)
        ax[1, index].plot(C)
        ax[1, index].set_xlabel('Terms', fontsize = 12)  
        ax[1, index].set_ylabel('Score', fontsize = 12)  

    ax[0, 0].title.set_text('Saliency Score DB1')
    ax[0, 1].title.set_text('Saliency Score DB2')
    ax[0, 2].title.set_text('Saliency Score DB3')

    ax[1, 0].title.set_text('Rank-k approximation DB1')
    ax[1, 1].title.set_text('Rank-k approximation DB2')
    ax[1, 2].title.set_text('Rank-k approximation DB3')

    plt.savefig('./plot.png')


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

    print(doc.terms[key_terms][:10])
    print("\n")
    
    for (i, sentence) in enumerate(key_sencences_sorted[:5]):
        print(i, ":", doc.raw_sentences[sentence])

    print("\n========= Key sentences ============\n")

    for (i, sentence) in enumerate(key_sencences):
        print(i, ":", doc.raw_sentences[sentence])
    
    return u, C

if __name__ == '__main__':
    for path in paths:
        print(f'Running algorithm on {path}\n')
        main(path)

    create_graph()