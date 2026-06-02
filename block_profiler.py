import re

class BlockProfiler:
    def __init__(self, triggers):
        # 'triggers' is a list of phonetic substrings/letters the user struggles with
        self.triggers = [t.lower() for t in triggers]

    def normalize_text(self, text):
        # Strip punctuation but keep actual words
        return re.sub(r'[^\w\s]', '', text.lower())

    def flag_problematic_words(self, text):
        clean_text = self.normalize_text(text)
        words = clean_text.split()
        
        flagged_words = set() # Using a set to avoid duplicates
        
        for word in words:
            for trigger in self.triggers:
                # For Phase 1, we are looking at hard starts (e.g., plosives at the beginning of a word)
                if word.startswith(trigger):
                    flagged_words.add(word)
                    break # Move to next word once flagged
                    
        return list(flagged_words)

if __name__ == "__main__":
    print("--- Phonetic: Block Profiler Initialization ---")
    
    # Simulating a user who struggles with hard 'p', 'b', and 'st' sounds
    user_triggers = ['p', 'b', 'st']
    profiler = BlockProfiler(triggers=user_triggers)
    
    # A sample script the user wants to read out loud
    test_script = "I am planning a big presentation and will start by explaining the baseline."
    
    print(f"Trigger Sounds: {user_triggers}")
    print(f"\nInput Script: '{test_script}'")
    
    flagged = profiler.flag_problematic_words(test_script)
    print(f"\nFlagged Words for Replacement: {flagged}")