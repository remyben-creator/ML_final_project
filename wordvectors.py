import spacy
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk



df = pd.read_csv('sentiments_grouped.csv')

# #preprocess text column
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# df['processed_sentiment'] = df['sentiment'].str.lower().apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words and word not in string.punctuation))

# df = df.drop('sentiment_group', axis=1)
# df.to_csv('sentiments_grouped.csv', index=False)

# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')

# process a sentence using the model
# categories = ["negative", "positive", "neutral", "angry", "happy", "sad", "fearful", "disgusted", "surprised"]
categories = ["negative", "positive", "happy", "fearful", "surprised"]
test_word = nlp("wow")

# Get the vector for each word
word_vectors = [nlp(word).vector for word in categories]

#find the category
most_similar_category = max(categories, key=lambda category: nlp(category).similarity(test_word))

print(most_similar_category)

def find_most_similar_category(row):
    test_word = nlp(row['processed_sentiment'])
    if not test_word:
        print(f"'{row['processed_sentiment']}' does not have a vector.")
        return None
    else:
        most_similar_category = max(categories, key=lambda category: nlp(category).similarity(test_word))
        return most_similar_category

df['sentiment_group'] = df.apply(find_most_similar_category, axis=1)
df.to_csv('sentiments_grouped.csv', index=False)

print(df['sentiment_group'].value_counts(dropna=False))

# sentiment_group
# positive     173
# negative     162
# happy        136
# fearful       93
# neutral       42
# disgusted     39
# sad           38
# surprised     37
# angry         12
# Name: count, dtype: int64

# sentiment_group
# negative    203
# positive    191
# happy       180
# sad         114
# neutral      44

# sentiment_group
# negative     205
# positive     177
# happy        159
# sad          100
# surprised     91

# sentiment_group
# negative     183
# positive     177
# happy        167
# fearful      135
# surprised     70