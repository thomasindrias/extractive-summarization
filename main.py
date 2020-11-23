from preprocessing.term_sentence import create_term_sentence

# Read and parse a text document, in this case an article about the Covid-19 vaccine
with open('data/wikipedia_text_mining.txt') as file:
    A = create_term_sentence(file)


# print(f'Number of terms: {len(terms)}')
# print(f'Number of sentences: {len(sentences)}')
# print(f'Maximum term frequency in one sentence: {np.max(A)}')

# print("\nTerms:")
# print(terms)

print("\nFirst column of A:")
print(A[:, 0])