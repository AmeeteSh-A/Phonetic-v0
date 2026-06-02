import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Importing the prep functions from your first file
from data_prep import get_synthetic_data, normalize_text

class PhoneticModel:
    def __init__(self):
        # ngram_range=(1,2) captures single words and pairs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        # Logistic Regression acting as our statistical proxy
        self.classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.classes_ = None

    def train(self, df):
        print("Training the statistical engine...")
        X = self.vectorizer.fit_transform(df['cleaned_input'])
        y = df['intent']
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        print("Training complete!")

    def predict_intent(self, text):
        cleaned = normalize_text(text)
        X_input = self.vectorizer.transform([cleaned])
        
        # We need probabilities for the symbolic layer to use later
        probs = self.classifier.predict_proba(X_input)[0]
        
        # Grab the highest probability and its corresponding intent
        best_idx = probs.argmax()
        best_intent = self.classes_[best_idx]
        confidence = probs[best_idx]
        
        return best_intent, confidence

if __name__ == "__main__":
    print("--- Phonetic: ML Engine Initialization ---")
    
    # 1. Load and prep data using your existing data_prep.py logic
    df = get_synthetic_data()
    df['cleaned_input'] = df['raw_input'].apply(normalize_text)
    
    # 2. Initialize and train model
    model = PhoneticModel()
    model.train(df)
    
    # 3. Test new, unseen fragmented inputs
    test_inputs = ["wtr pls", "need f00d", "thnk u", "hurt lg"]
    print("\nTesting untrained fragmented inputs:")
    for text in test_inputs:
        intent, conf = model.predict_intent(text)
        print(f"Input: '{text}' -> Predicted Intent: '{intent}' (Confidence: {conf:.2f})")