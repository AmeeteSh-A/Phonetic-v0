import os
import re
import random
import nltk
from nltk.corpus import wordnet, webtext
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from lemminflect import getInflection

# 1. Ensure all required NLTK academic libraries are present
for resource in ['punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'webtext', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}' if resource in ['wordnet', 'webtext'] else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class InteractivePhonetic:
    def __init__(self):
        self.triggers = set()
        self.lemmatizer = WordNetLemmatizer()
        
        print("Booting up the Neurosymbolic Engine...")
        print("Training Context Scorer on the WebText Corpus (please wait a few seconds)...")
        self.vectorizer = CountVectorizer(ngram_range=(2, 3))
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._train_scorer()
        os.system('cls' if os.name == 'nt' else 'clear')

    def _train_scorer(self, num_sentences=5000):
        # Trains the ML model to understand standard English grammar using modern conversational data
        real_sentences = [' '.join(sent).lower() for sent in webtext.sents()[:num_sentences]]
        words = [word.lower() for word in webtext.words()[:num_sentences * 10] if word.isalpha()]
        fake_sentences = [' '.join(random.sample(words, k=random.randint(5, 10))) for _ in range(num_sentences)]
        
        X_text = real_sentences + fake_sentences
        y = [1] * len(real_sentences) + [0] * len(fake_sentences)
        
        X = self.vectorizer.fit_transform(X_text)
        self.classifier.fit(X, y)

    def _get_wordnet_pos(self, treebank_tag):
        # Maps NLTK's detailed POS tags to WordNet's broader categories
        if treebank_tag.startswith('J'): return wordnet.ADJ
        elif treebank_tag.startswith('V'): return wordnet.VERB
        elif treebank_tag.startswith('N'): return wordnet.NOUN
        elif treebank_tag.startswith('R'): return wordnet.ADV
        else: return None

    def _extract_onset(self, word):
        # Extracts the starting consonant or consonant cluster (e.g., 'pl' from 'planning', 'b' from 'big')
        match = re.match(r'^([^aeiou]+)', word, re.IGNORECASE)
        return match.group(1).lower() if match else word[0].lower()

    def run_diagnostic(self):
        print("="*60)
        print("PHASE 1: PHONETIC DIAGNOSTIC")
        print("="*60)
        diagnostic_text = (
            "Please read the following paragraph aloud:\n\n"
            "'The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
            "liquor jugs. A brave strong cat drank fresh milk. Grumpy wizards make toxic brew. "
            "She sells seashells by the seashore. Peter Piper picked a peck of pickled peppers.'"
        )
        print(diagnostic_text)
        print("\nType the words from the paragraph that caused a stammer or block.")
        print("Separate them with spaces (e.g., 'pack box jumps'):")
        
        difficult_words = input("> ").strip().split()
        
        # Dynamically build the trigger profile based on the starting sounds
        for word in difficult_words:
            onset = self._extract_onset(word)
            self.triggers.add(onset)
            
        triggers_list = list(self.triggers)
        print(f"\n=> Profile Created! Your dynamic phonetic triggers are: {triggers_list}")

        # Unsupervised ML: K-Means Clustering on Phonetic Features
        if len(triggers_list) >= 3:
            print("\nRunning K-Means Clustering to analyze phonetic patterns...")
            # We use character-level vectors to group sounds mathematically
            char_vec = CountVectorizer(analyzer='char')
            X_cluster = char_vec.fit_transform(triggers_list)
            
            num_clusters = 2
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(X_cluster)
            
            clusters = {i: [] for i in range(num_clusters)}
            for i, label in enumerate(kmeans.labels_):
                clusters[label].append(triggers_list[i])
                
            for cluster_id, items in clusters.items():
                print(f" - Phonetic Cluster {cluster_id + 1}: {items}")
        print("\n")

    def get_safe_synonyms(self, word, pos_tag):
        wn_pos = self._get_wordnet_pos(pos_tag)
        candidates = set()
        
        # We lemmatize the word so WordNet can find it easily
        lemma_form = self.lemmatizer.lemmatize(word, pos=wn_pos) if wn_pos else word
        
        # Fetch synonyms strictly matching the part of speech (Verb for Verb, etc.)
        synsets = wordnet.synsets(lemma_form, pos=wn_pos) if wn_pos else wordnet.synsets(lemma_form)
        
        for syn in synsets:
            for lemma in syn.lemmas():
                candidate = lemma.name().lower().replace('_', ' ')
                candidates.add(candidate)
                
        # Filter out the original word and any words starting with trigger sounds
        safe_candidates = []
        for cand in candidates:
            if cand != word and not any(cand.startswith(t) for t in self.triggers):
                safe_candidates.append(cand)
                
        return safe_candidates

    def interactive_loop(self):
        print("="*60)
        print("PHASE 2: REAL-TIME ASSISTANT")
        print("="*60)
        print("Type any sentence you want to say. The engine will rewrite it to be safe.")
        print("Type 'quit' to exit.\n")
        
        while True:
            text = input("You: ")
            if text.lower() == 'quit':
                break
                
            # Tokenize and tag the parts of speech
            words = word_tokenize(text)
            tagged_words = nltk.pos_tag(words)
            
            safe_words = words.copy()
            replacements_made = False
            trigger_hit = False
            
            for i, (word, tag) in enumerate(tagged_words):
                if not word.isalpha(): continue # Skip punctuation
                
                # Check if word starts with any of the user's triggers
                if any(word.lower().startswith(t) for t in self.triggers):
                    trigger_hit = True
                    
                    candidates = self.get_safe_synonyms(word.lower(), tag)
                    
                    if not candidates:
                        # Graceful Fallback: Just keep the original word if no safe synonym exists
                        continue
                        
                    # Create a template for the ML model (e.g., "I need a ___ job")
                    template_list = safe_words.copy()
                    template_list[i] = "___"
                    template_str = ' '.join(template_list)
                    
                    best_candidate = None
                    highest_score = -1.0
                    
                    # Score candidates using context grammar
                    for cand in candidates:
                        test_phrase = template_str.replace("___", cand)
                        try:
                            X_test = self.vectorizer.transform([test_phrase])
                            score = self.classifier.predict_proba(X_test)[0][1]
                        except:
                            score = 0.0
                            
                        # 40% Confidence Threshold Check
                        if score > highest_score and score > 0.40:
                            highest_score = score
                            best_candidate = cand
                            
                    if best_candidate:
                        # NLP Morphology Engine: Conjugate the base synonym to match the original word's tense/plurality
                        if tag.startswith(('N', 'V', 'J', 'R')):
                            inflected_options = getInflection(best_candidate, tag)
                            final_word = inflected_options[0] if inflected_options else best_candidate
                        else:
                            final_word = best_candidate
                        
                        safe_words[i] = final_word
                        replacements_made = True
            
            # Reconstruct the sentence beautifully (fixing punctuation spacing)
            final_sentence = ' '.join(safe_words)
            final_sentence = re.sub(r'\s+([?.!,;\'])', r'\1', final_sentence)
            
            if replacements_made:
                print(f"Safe Script: {final_sentence}\n")
            elif trigger_hit:
                print(f"Safe Script: (Triggers caught, but no safe/confident synonyms found) {text}\n")
            else:
                print(f"Safe Script: (No triggers detected) {text}\n")
                
            # --- CONTINUAL LEARNING LOOP ---
            feedback = input("Did you stammer on any words? (Type them, or press Enter to skip): ").strip()
            
            if feedback.lower() == 'quit':
                break
            elif feedback:
                new_triggers = False
                for w in feedback.split():
                    onset = self._extract_onset(w)
                    if onset not in self.triggers:
                        self.triggers.add(onset)
                        new_triggers = True
                
                if new_triggers:
                    print(f"=> Profile Updated! Your active triggers are now: {list(self.triggers)}\n")
            else:
                print()
            # ------------------------------------

if __name__ == "__main__":
    app = InteractivePhonetic()
    app.run_diagnostic()
    app.interactive_loop()