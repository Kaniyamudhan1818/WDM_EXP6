### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 24-11-2026
### NAME: KANIYAMUDHAN V
### REG NO: 212224040148
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:


import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))

    tokens = [
        token for token in tokens
        if token not in stop_words
        and token not in string.punctuation
    ]

    return " ".join(tokens)

# Preprocess documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(
    preprocessed_docs.values()
)

# Search function
def search(query):

    # Preprocess query
    preprocessed_query = preprocess_text(query)

    # Convert query into TF-IDF vector
    query_vector = tfidf_vectorizer.transform(
        [preprocessed_query]
    )

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Store search results
    results = []

    for i, score in enumerate(cosine_similarities):

        doc_id = list(documents.keys())[i]
        document = documents[doc_id]

        results.append((doc_id, document, score))

    # Sort by similarity score
    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return results

# Get query from user
query = input("Enter your query: ")

# Perform search
search_results = search(query)

# Display results
print("\nQuery:", query)
print("\nSearch Results:")

for i, result in enumerate(search_results, start=1):

    print("\nRank:", i)
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", round(result[2], 4))
    print("----------------------")

# Display highest cosine similarity score
if search_results:

    highest_rank_score = search_results[0][2]

    print(
        "\nThe highest rank cosine score is:",
        round(highest_rank_score, 4)
    )

else:
    print("No results found.")


### Output:
<img width="853" height="690" alt="image" src="https://github.com/user-attachments/assets/daa02640-2c62-40ca-987b-c2e7e4800772" />


### Result:
Thus, the TF-IDF model and Cosine Similarity-based Information Retrieval system were successfully implemented using Python. The documents were preprocessed, similarity scores were calculated, and the relevant documents were retrieved and ranked according to their similarity scores.
