import gensim

sentences=[
    "ram is a boy".split(),
    "pooja is a girl".split(),
    "sita is a girl".split(),
]

vocab=[
    "ram boy".split(),
    "pooja girl".split(),
    "sita".split(),
]

model = gensim.models.word2vec.Word2Vec(vector_size=2, min_count=1)
model.build_vocab(vocab)
model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=model.epochs)
print(model.wv.similar_by_word("boy"))
