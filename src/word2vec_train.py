from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Sample corpus: list of sentences
corpus = [
    "Natural language processing enables computers to understand human language.",
    "Word embeddings are a type of word representation that allows words to be represented as vectors.",
    "Gensim is a popular library for unsupervised topic modeling and natural language processing.",
    "Word2Vec is an algorithm for learning vector representations of words.",
    "Machine learning and deep learning are subfields of artificial intelligence."
]

# Tokenize the corpus
sentences = [nltk.word_tokenize(sent.lower()) for sent in corpus]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2, sg=1)

# Save the model
model.save("word2vec_sample.model")

# Example usage: find most similar words
target_word = "language"
if target_word in model.wv:
    print(f"Most similar words to '{target_word}':")
    for word, score in model.wv.most_similar(target_word, topn=5):
        print(f"  {word}: {score:.3f}")
else:
    print(f"'{target_word}' not in vocabulary.")

# Example: get vector for a word
print(f"\nVector for 'language':\n{model.wv['language']}") 