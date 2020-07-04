# N-gram models for language modeling

We simply built n-gram models which are simple baseline language models. More precisely we built **bigram** and **trigram** models. Raw n-gram models cannot be evaluated on a test set as they overfit training text data too much. Hence it is required to use a **smoothing** technique (regularization). We chose an intuitive one by computing the **convex mixture** between an ngram model and the **uniform** distribution over vocabulary as the convex mixture of probability distributions remain a probability distribution.

As for dataset we simply used the book **Pride and Prejudice** of Jane Austen which appears to be the most downloaded ebooks from [Gutenberg project](https://www.gutenberg.org/ebooks/search/%3Fsort_order%3Ddownloads). 

We then performed 2 tasks:
  - language model evaluation by computing **perplexity** on test set
  - text generation using raw bigram model

## Dataset

**Data**: ebook Pride and Prejudice by Jane Austen

**training**: 60 first chapters roughly 125 000 tokens

**test**: last chapter roughly 1250 tokens

## Preprocessing

raw data:
```
“My dear Mr. Bennet,” said his lady to him one day, “have you
      heard that Netherfield Park is let at last?”

      Mr. Bennet replied that he had not.
```

We only performed basic preprocessing as we want to be able to generate text. Although using stemming or/and lemmatization could have helped us to 'absorb' more words into our vocabulary.

training data:

```
<S> my dear mr bennet said his lady to him one day have you heard that netherfield park is let at last </S> <S> mr bennet replied that he had not </S>
```

