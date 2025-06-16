import streamlit as st
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
import os

MODEL_PATH = "word2vec_sample.model"

st.title("Word2Vec Explorer")

st.write("""
Enter your own text corpus, train a Word2Vec model, and explore word similarities interactively!
""")

# User input for corpus
def_corpus = """Natural language processing enables computers to understand human language.\nWord embeddings are a type of word representation that allows words to be represented as vectors.\nGensim is a popular library for unsupervised topic modeling and natural language processing.\nWord2Vec is an algorithm for learning vector representations of words.\nMachine learning and deep learning are subfields of artificial intelligence."""

user_corpus = st.text_area("Enter your corpus (one sentence per line):", value=def_corpus, height=150)

# User input for target word
target_word = st.text_input("Enter a target word to find similar words:", value="language")

# Number of similar words
topn = st.slider("Number of similar words to show:", min_value=1, max_value=10, value=5)

# Train model button
if st.button("Train & Explore"):
    # Tokenize
    sentences = [nltk.word_tokenize(sent.lower()) for sent in user_corpus.split('\n') if sent.strip()]
    # Train model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2, sg=1)
    # Save model
    model.save(MODEL_PATH)
    st.success("Model trained and saved!")
    # Show vocabulary
    st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")
    # Show similar words
    if target_word in model.wv:
        st.subheader(f"Most similar words to '{target_word}':")
        sim_words = model.wv.most_similar(target_word, topn=topn)
        st.table([{"Word": w, "Similarity": f"{s:.3f}"} for w, s in sim_words])
        # Show vector
        st.subheader(f"Embedding vector for '{target_word}':")
        st.write(model.wv[target_word])
    else:
        st.error(f"'{target_word}' not in vocabulary.")

# Optionally user can see last model result

if os.path.exists(MODEL_PATH):
    st.markdown("---")
    st.write("Or, query the last trained model:")
    query_word = st.text_input("Word to query in last trained model:", value="language", key="query_word")
    n_sim = st.slider("Number of similar words:", min_value=1, max_value=10, value=5, key="n_sim")
    if st.button("Query Model"):
        model = Word2Vec.load(MODEL_PATH)
        if query_word in model.wv:
            st.subheader(f"Most similar words to '{query_word}':")
            sim_words = model.wv.most_similar(query_word, topn=n_sim)
            st.table([{"Word": w, "Similarity": f"{s:.3f}"} for w, s in sim_words])
            st.subheader(f"Embedding vector for '{query_word}':")
            st.write(model.wv[query_word])
        else:
            st.error(f"'{query_word}' not in vocabulary.") 
