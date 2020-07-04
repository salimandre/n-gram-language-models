# N-gram models for language modeling

We simply built n-gram models which are simple baseline language models. More precisely we built **bigram** and **trigram** models. Raw n-gram cannot be evaluated as they overfit training text data too much. Hence it is required to use a **smoothing** technique (regularization). We chose an intuitive simple one by computing the **convex mixture** between an ngram model and the uniform distribution over vocabulary as the convex mixture of probability distributions remain a probability distribution.

https://www.gutenberg.org/ebooks/search/%3Fsort_order%3Ddownloads
