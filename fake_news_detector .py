import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import pickle
import os
import string



nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def train_and_save_model(csv_path='dataset/fake_or_real_news.csv', 
                        model_path='model.pkl', 
                        vector_path='vector.pkl'):
    try:
       
        df = pd.read_csv(
            csv_path,
            encoding='utf-8',
            on_bad_lines='warn',
            engine='python',
            quoting=0
        )
        
        
        if not {'text', 'label'}.issubset(df.columns):
            raise ValueError("CSV must contain 'text' and 'label' columns")

        df = df[['text', 'label']].dropna()

        print("Class distribution:\n", df['label'].value_counts())

        # Clean text
        df['cleaned_text'] = df['text'].apply(clean_text)

        
        df['label'] = df['label'].apply(lambda x: 1 if x.strip().upper() == 'FAKE' else 0)
        X = df['cleaned_text']
        y = df['label']
         
   

        
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=stopwords.words('english')
        )
        X_vect = tfidf.fit_transform(X)

        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vect, y, 
            test_size=0.25, 
            random_state=42,
            stratify=y
        )

        
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))

        
        model = LogisticRegression(
            class_weight=class_weights,
            max_iter=1000,
            solver='saga',
            verbose=1
        )
        model.fit(X_train, y_train)

        # hello sajeeb ahmed 


        
        y_pred = model.predict(X_test)
        print("\nEvaluation Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(vector_path, 'wb') as f:
            pickle.dump(tfidf, f)

        print("\nModel saved successfully.")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Possible solutions:")
        print("1. Verify CSV file exists at specified path")
        print("2. Check CSV file encoding (try UTF-8)")
        print("3. Ensure CSV has 'text' and 'label' columns")
        print("4. Remove any special characters from the CSV")

if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    train_and_save_model()