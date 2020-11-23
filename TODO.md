# TODO ✅

- Text preprocessing (Frans)
    - Lemmmatizing/stemming
    - Remove HTML and LaTeX tags
    - Remove stop-words and non-alphabetics (e.g. #"@9!)
    - Make sure the final list of terms consists of unique elements

- Form term-sentence matrix (Frans)
    - Use some kind of term frequency weighting (e.g. TF-IDF)
    - Maybe consider some way of representing sparse matrices

- Compute saliency scores from the term-sentence matrix (Fredrik)
    - SVD in NumPy or maybe we have to code it ourselves?
    - Keyword extraction
    - Also key sentence, but how?

- Key sentence extraction using a rank-k approximation (Thomas)
    - Builds on the first method?
    - More research is needed!