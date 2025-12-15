from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Step 1: Documents (2 Sports + 2 Politics)
sports_docs = [
    "The football team won the championship after a thrilling match.",
    "The basketball player scored a record number of points this season."
]

politics_docs = [
    "The government passed a new policy to improve the education system.",
    "The election campaign focused on economic growth and healthcare reforms."
]

# Combine all documents
documents = sports_docs + politics_docs

# Step 2: Bag of Words (BoW)
bow = CountVectorizer()
bow_matrix = bow.fit_transform(documents)

print("Bag of Words Matrix:\n", bow_matrix.toarray())
print("Feature Names (Words):\n", bow.get_feature_names_out())

# Step 3: TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())
print("Feature Names (Words):\n", tfidf.get_feature_names_out())
