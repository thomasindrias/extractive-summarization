import spacy
import numpy as np
from preprocessing.document import Document
from preprocessing.term_sentence import create_term_sentence

# English language processing object
nlp = spacy.load('en_core_web_sm')

# Read and parse a text document, in this case an article about the Covid-19 vaccine
doc = Document('data/wikipedia_text_mining.txt', nlp)

# Create a term-sentence matrix from the document content
A = create_term_sentence(doc.terms, doc.sentences)


print(f'Number of terms: {len(doc.terms)}')
print(f'Number of sentences: {len(doc.sentences)}')
print(f'Maximum term frequency in one sentence: {np.max(A)}')

print("\nTerms:")
print(doc.terms)

print("\nFirst column of A:")
print(A[:, 0])