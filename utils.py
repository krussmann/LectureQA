import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import numpy as np


def wordnetmap(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV

def preprocess(text, style="None"):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in string.punctuation]
#     tokens = [re.sub("[^a-zA-Z]+", "", word) for word in tokens] # hard
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    if style == "lem":
        # Create tag with kind of word e.g. ('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ')
        tagged = pos_tag(tokens)
        # Convert POS tag from Penn tagset to WordNet tagset and filter out the words without match (if there are any)
        tokenized_and_tagged = [(word, wordnetmap(tag)) for word, tag in tagged if wordnetmap(tag) is not None]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word=word, pos=tag) for word, tag in tokenized_and_tagged]
    elif style == "stem":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


# Function to aggregate word embeddings over sentence
def compute_sentence_embedding(sentence, word_vectors, bigrams_m="none"):
    # Helper function to generate bigrams from a list of words
    def generate_bigrams(words):
        return [f'{words[i]}_{words[i + 1]}' for i in range(len(words) - 2)]

    # Tokenize the sentence into words
    words = sentence.split()
    bigrams = generate_bigrams(words)

    embeddings = []
    if bigrams_m != "only":
        for word in words:
            # Check if word exists in the vocabulary
            if word in word_vectors:
                # Retrieve word embedding vector
                embeddings.append(word_vectors[word])
    if bigrams_m != "none":
        for bigram in bigrams:
            # Check if bigram exists in the vocabulary
            if bigram in word_vectors:
                # Retrieve bigram embedding vector
                embeddings.append(word_vectors[bigram])
    if embeddings:
        # Aggregate word embeddings using mean or sum
        # return np.mean(embeddings, axis=0)  # Change to np.sum for sum aggregation
        return np.sum(embeddings, axis=0)
    else:
        # Return zero vector if no word embeddings found
        return np.zeros(word_vectors.vector_size)