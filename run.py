import requests
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Define your API key and headers
api_key = 'YOUR_API_KEY'
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Define the ID of the lead you want to fetch
lead_id = 'lead_id'  # replace with your lead_id

# Make the GET request
response = requests.get(f'https://api.close.com/api/v1/lead/{lead_id}', headers=headers)

# Parse the response
data = response.json()

# Extract the activities
activities = data['activities']

# Initialize a CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the data and transform the data
X = vectorizer.fit_transform(activities)

# Now X is a matrix where each row represents an activity and each column represents a word. 
# The value in each cell is the count of the word in the corresponding activity.

# If you want to use TF-IDF instead of counts, you can use TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(activities)

# Now X_tfidf is a matrix where each row represents an activity and each column represents a word. 
# The value in each cell is the TF-IDF score of the word in the corresponding activity.