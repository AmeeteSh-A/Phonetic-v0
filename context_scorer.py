import random
import nltk
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure the corpus is downloaded on the machine
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')

class ContextScorer:
    def __init__(self):
        # We use word n-grams (2 to 3 words) to capture context 
        self.vectorizer = CountVectorizer(ngram_range=(2, 3))
        # Increased max_iter because we are training on thousands of sentences now
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        
    def train_on_brown_corpus(self, num_sentences=5000):
        print(f"Fetching {num_sentences} sentences from the Brown Corpus...")
        
        # 1. POSITIVE CLASS (1): Real English phrasing
        # Join the tokenized words back into continuous string sentences
        real_sentences = [' '.join(sent).lower() for sent in brown.sents()[:num_sentences]]
        
        # 2. NEGATIVE CLASS (0): Unnatural English phrasing
        # We grab a large pool of words and create random "nonsense" sentences
        words = [word.lower() for word in brown.words()[:num_sentences * 10] if word.isalpha()]
        fake_sentences = []
        for _ in range(num_sentences):
            # Create a fake sentence of random length between 5 and 10 words
            fake_sentence = ' '.join(random.sample(words, k=random.randint(5, 10)))
            fake_sentences.append(fake_sentence)
            
        print("Vectorizing and training the statistical model (this may take a few seconds)...")
        X_text = real_sentences + fake_sentences
        y = [1] * len(real_sentences) + [0] * len(fake_sentences)
        
        # Train the model on the combined real vs. fake dataset
        X = self.vectorizer.fit_transform(X_text)
        self.classifier.fit(X, y)
        print("Training complete! The model now understands standard English syntax.")

    def score_candidates(self, sentence_template, candidates):
        best_candidate = None
        highest_score = -1.0
        
        print(f"\nEvaluating candidates for template: '{sentence_template}'")
        
        for word in candidates:
            test_phrase = sentence_template.replace("___", word)
            
            try:
                X_test = self.vectorizer.transform([test_phrase])
                score = self.classifier.predict_proba(X_test)[0][1] 
            except Exception:
                score = 0.0
                
            print(f" - '{word}': {score:.4f}")
            
            if score > highest_score:
                highest_score = score
                best_candidate = word
                
        return best_candidate

if __name__ == "__main__":
    print("--- Phonetic: Real-World Context Scorer Initialization ---")
    
    scorer = ContextScorer()
    # Training on 5000 real sentences and 5000 fake sentences
    scorer.train_on_brown_corpus(num_sentences=5000)
    
    template = "design a ___ presentation"
    candidates = ['great', 'handsome', 'grown', 'enceinte']
    
    best_word = scorer.score_candidates(template, candidates)
    print(f"\n=> Contextually Best Replacement: '{best_word}'")