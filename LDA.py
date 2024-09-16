import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Save the DataFrame to a CSV file
output_csv = 'BBC_News_Article.csv'

# Load the data from CSV
df = pd.read_csv(output_csv)

# Display the first few rows to understand the structure of your data
print("Overview of the few row")
print(df)

# Preprocess the text data
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    return tokens

# Apply preprocessing to each row in the DataFrame
df['processed_text'] = df['description'].apply(preprocess_text)

# Check the processed text
print("Processed Text")
print(df['processed_text'])

# Create a Gensim Dictionary and Corpus
dictionary = corpora.Dictionary(df['processed_text'])


# Filter out tokens that appear in less than 3 documents or more than 50% of the documents
dictionary.filter_extremes(no_below=3, no_above=0.5)

# Check the dictionary after filtering (print only first 50 key-value pairs)
print("Dictionary AFTER Filter (first 50 tokens):")
print(dict(list(dictionary.token2id.items())[:50]))

# Convert each document into the bag-of-words (BoW) format
corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]

# Build and Train LDA Model
num_topics = 5
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# Assign Topics to Each Document
topics = []
for text in range(len(df['processed_text'])):
    topic_distribution = lda_model[corpus[text]]
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    topics.append(sorted_topics[0][0])

df['topic'] = topics

# Count Topic Occurrences
topic_counts = df['topic'].value_counts()

# Print Topic Counts
print("Topic Counts:")
print(topic_counts)

# Optionally, you can print the topics themselves
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics():
    if idx > 0:
        print()  # Add a blank line before printing topics after the first one
    print(f"Topic {idx}: {topic}")


# Convert LDA topic distribution into document-topic vectors
def lda_to_document_topic_vectors(lda_model, corpus):
    """ Convert LDA model and corpus into document-topic vectors """
    document_vectors = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        document_vectors.append([topic[1] for topic in topic_dist])
    return np.array(document_vectors)

# Convert LDA topic embeddings into topic-word vectors
def lda_to_topic_vectors(lda_model):
    """ Convert LDA topics into topic-word vectors (enhanced topic vectors) """
    topic_vectors = []
    for topic_id in range(lda_model.num_topics):
        topic = lda_model.get_topic_terms(topic_id, topn=len(dictionary))
        vector = np.zeros(len(dictionary))
        for word_id, prob in topic:
            vector[word_id] = prob
        topic_vectors.append(vector)
    return np.array(topic_vectors)

# Get document-topic vectors and topic-word vectors
document_vectors = lda_to_document_topic_vectors(lda_model, corpus)
enhanced_topic_vectors = lda_to_topic_vectors(lda_model)

# Apply PCA for dimensionality reduction on document-topic vectors (5-dimensional to 2D)
pca_documents = PCA(n_components=2)
traditional_document_embeddings_2d = pca_documents.fit_transform(document_vectors)

# Apply PCA for dimensionality reduction on topic-word vectors (13,487-dimensional to 2D)
pca_topics = PCA(n_components=2)
traditional_topic_embeddings_2d = pca_topics.fit_transform(enhanced_topic_vectors)

# Plot the PCA results with both document and topic embeddings
plt.figure(figsize=(10, 7))
plt.scatter(traditional_document_embeddings_2d[:, 0], traditional_document_embeddings_2d[:, 1], c=df['topic'], cmap='rainbow', alpha=0.7, label='Documents')
plt.scatter(traditional_topic_embeddings_2d[:, 0], traditional_topic_embeddings_2d[:, 1], c='black', marker='X', s=200, label='Topics')
plt.title("Traditional LDA Topic Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Topic")
plt.legend()
plt.grid(True)



# Prepare and Display pyLDAvis Visualization
lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)


print()

pyLDAvis.save_html(lda_display, 'Traditional_LDA_Visualization.html')
print("LDA visualization has been saved to Traditional_LDA_Visualization.html")

print()

# Save the PCA visualization
plt.savefig('Traditional_LDA_Visualization_PCA.png')
plt.show()

print("PCA visualization saved as 'Traditional_LDA_Visualization_PCA.png'")
