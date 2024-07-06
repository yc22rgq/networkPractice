import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_data['Text'])

le = LabelEncoder()
y = le.fit_transform(train_data['Label'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

print(rf_model.score(X_val, y_val))
