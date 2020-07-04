import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string
import nltk
nltk.download('punkt')
from collections import Counter


#----- Preprocessing tools -----


def preprocess(input_text):

    input_text = input_text.lower()
    input_text = input_text.translate(str.maketrans('', '', string.punctuation+'“'+'”'))
    return input_text

def make_vocab(text_input,vocab_size, show_plot=False):
    
    # sort words by count
    all_words=text_input.split()
    count_total=len(all_words)
    sorted_words=Counter(all_words).most_common()
        
    # show word distribution
    if show_plot:
        histo = sns.barplot(list(range(1,len(sorted_words[:vocab_size])+1)), [value for key, value in sorted_words[:vocab_size]])
        plt.xticks(np.linspace(1,len(sorted_words[:vocab_size]),10).astype(int) ,np.linspace(1,len(sorted_words[:vocab_size]),10).astype(int))
        plt.title('vocabulary words distribution')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()
    
    # keep first dico_size words as vocab
    list_in_vocab=[w for w,c in sorted_words[:vocab_size]]
    list_out_vocab=[w for w,c in sorted_words[vocab_size:]]
    count_in_vocab=[c for w,c in sorted_words[:vocab_size]]

    # in vocab
    vocab = {w: i+1 for i, w in enumerate(list_in_vocab)}
    vocab['<OOV>']=0
    vocab['<S>']=-1
    vocab['</S>']=-2
    # out of vocab
    out_of_vocab={}
    for w in list_out_vocab:
        out_of_vocab[w]=0
    
    print('\nnb total words: ',count_total)
    print('nb total distinct words: ',len(sorted_words))
    print('dico size: ',vocab_size)
    print('out of vocab (OOV): ',round(100*(count_total-sum(count_in_vocab))/count_total,1),'%\n')
    return vocab, out_of_vocab

def tokenize(text_input, oov_list):
    tokenized_text=''
    for sentence in text_input.split('\n\n'):
        tokenized_sentence='<S> '
        for word in sentence.split():
            if word not in oov_list:
                tokenized_sentence+=word+' '
            else:
                tokenized_sentence+='<OOV> '
        tokenized_sentence+='</S> '
        tokenized_text += tokenized_sentence

    return tokenized_text


#----- Language Modeling tools-----


class LanguageModel:
    def __init__(self):
        pass

    def proba(self, past_words, target):
        pass


class UniformLM(LanguageModel):
    def __init__(self,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size

    def proba(self, past_words, target):
        return 1./self.vocab_size    


class Ngram(LanguageModel):
    def __init__(self, n):
        super().__init__()
        self.n=n
        self.counts={}
        self.lm={}

    def __make_key(self,list_word):
        key=''
        for w in list_word:
            key+=w+' '
        return key.strip()

    def count(self, text_input):
        
        data_ngrams = [(tuple(x[:self.n-1]), x[self.n-1]) for x in nltk.ngrams(text_input.split(), self.n)]

        for ng,target in data_ngrams:
            
            
            key_ng = self.__make_key(ng)

            try:
                self.counts[key_ng]
            except KeyError:
                self.counts[key_ng]={}

            try:
                self.counts[key_ng][target]+=1
            except KeyError:
                self.counts[key_ng][target]=1
           
    def make_lm(self, vocabulary, smoothing_lm=None, mixture_param=0.5):

        if smoothing_lm is not None:        
            for ng in self.counts.keys():
                self.lm[ng]={}
                counts_ng=sum(self.counts[ng].values())
                for w in vocabulary:
                        try:
                            self.lm[ng][w] = (1. - mixture_param) * self.counts[ng][w] / counts_ng + mixture_param * smoothing_lm.proba(ng.split(),w)
                        except KeyError:
                            self.lm[ng][w] = mixture_param * smoothing_lm.proba(ng.split(),w)

        else:
            for ng in self.counts.keys():
                self.lm[ng]={}
                counts_ng=sum(self.counts[ng].values())
                for w in vocabulary:
                        try:
                            self.lm[ng][w] = self.counts[ng][w] / counts_ng
                        except KeyError:
                            self.lm[ng][w] = 0.       

    def proba(self, past_words, target):
        past_words=past_words[-self.n+1:]
        key_ng = self.__make_key(past_words)
        return self.lm[key_ng][target]

    def predict(self,ngrams_list,target):
        
        ngrams_list=ngrams_list[-self.n+1:]
        key_ng=''
        for w in ngrams_list:
            key_ng+=w+' '
        return self.lm[key_ng.strip()][target]
    
    def evaluate(self, text_input):
        
        perplexity=0
        cross_entropy=0
        
        data_ngrams = [(tuple(x[:self.n-1]), x[self.n-1]) for x in nltk.ngrams(text_input.split(), self.n)]
        n_evals = len(data_ngrams)
        
        for ng,target in data_ngrams:
            try:
                proba=self.predict(ng,target)
            except KeyError:
                proba=1.
                n_evals-=1
            cross_entropy+=-np.log2(proba)
        
        perplexity = 2**(cross_entropy / n_evals)
        
        return perplexity
        
    def generate(self, n_samples):
        
        if self.n==2:
            key_ng='<S>'
        elif self.n==3:
            key_ng='</S> <S>'
        else:
            key_ng = random.choice(list(self.lm.keys()))

        for i in range(n_samples-2):

            new_word = np.random.choice(list(self.lm[key_ng].keys()), 1, p=list(self.lm[key_ng].values()))[0]

            key_split=key_ng.split()[1:]+[new_word]
            key_ng = self.__make_key(key_split) 
            yield new_word




