import nltk
from nltk.corpus import wordnet

# Download the WordNet database if it's not already on your machine
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

class SynonymEngine:
    def __init__(self, triggers):
        self.triggers = [t.lower() for t in triggers]

    def get_safe_synonyms(self, word):
        candidates = set()
        
        # Fetch synonyms from WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                candidate_word = lemma.name().lower()
                # Replace underscores with spaces for multi-word synonyms
                candidate_word = candidate_word.replace('_', ' ')
                candidates.add(candidate_word)
                
        # Filter out the original word itself
        if word in candidates:
            candidates.remove(word)
            
        # Filter out any synonyms that start with the user's phonetic triggers
        safe_synonyms = []
        for candidate in candidates:
            starts_with_trigger = any(candidate.startswith(t) for t in self.triggers)
            if not starts_with_trigger:
                safe_synonyms.append(candidate)
                
        return safe_synonyms

if __name__ == "__main__":
    print("--- Phonetic: Synonym Engine Initialization ---")
    
    # Using the same triggers from our profiler
    user_triggers = ['p', 'b', 'st']
    engine = SynonymEngine(triggers=user_triggers)
    
    # Testing with the flagged words from your last output
    flagged_words = ['baseline', 'big', 'planning', 'start', 'presentation']
    
    for word in flagged_words:
        safe_options = engine.get_safe_synonyms(word)
        print(f"\nOriginal Word: '{word}'")
        if safe_options:
            print(f"Safe Alternatives: {safe_options[:5]}") # Showing top 5 for brevity
        else:
            print("Safe Alternatives: None found in WordNet")