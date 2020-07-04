from ngrams_toolbox import *

# loading text files
PATH_DATA='/Users/mac/Desktop/python/language_modeling/data/'
file = open(PATH_DATA+'pride_and_prejudice_train.txt', 'r') 
train_raw = file.read()
file = open(PATH_DATA+'pride_and_prejudice_test.txt', 'r') 
test_raw = file.read()

# build vocab
vocab, out_of_vocab = make_vocab(text_input=preprocess(train_raw+' '+test_raw), vocab_size=1000, show_plot=False)
index_to_token = {v: k for k, v in vocab.items()}

# preprocess then tokenize
train_final = tokenize(preprocess(train_raw), out_of_vocab.keys())
test_final = tokenize(preprocess(test_raw), out_of_vocab.keys())

'''
# initialize lm models

bigram_lm = Ngram(2)
bigram_lm.count(train_final)

trigram_lm = Ngram(3)
trigram_lm.count(train_final)

unif_lm = UniformLM(len(vocab))

# smoothed bigrams

bigram_lm.make_lm(vocab, smoothing_lm=unif_lm, mixture_param=.2)

print('--- smoothed bigram ---')
print('train ppl: ',bigram_lm.evaluate(train_final))
print('test ppl: ',bigram_lm.evaluate(test_final),'\n')

# smoothed trigrams

trigram_lm.make_lm(vocab, smoothing_lm=unif_lm, mixture_param=0.5)

print('--- smoothed trigrams ---')
print('train ppl: ',trigram_lm.evaluate(train_final))
print('test ppl: ',trigram_lm.evaluate(test_final),'\n')

# bigram/trigram mixture

trigram_lm.make_lm(vocab, smoothing_lm=bigram_lm, mixture_param=0.8)

print('--- trigram/bigram mixture ---')
print('train ppl: ',trigram_lm.evaluate(train_final))
print('test ppl: ',trigram_lm.evaluate(test_final),'\n')
'''

# generating text

gen_lm = Ngram(3)
gen_lm.count(train_final)

gen_lm.make_lm(vocab, smoothing_lm=None)

for i in range(10):
	print('example of generated text: \n')
	for w in gen_lm.generate(50):
		print(w, end=" ")
	print('\n')























