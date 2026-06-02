import re
from block_profiler import BlockProfiler
from synonym_engine import SynonymEngine
from context_scorer import ContextScorer
from pangram_assessment import PangramAssessor

class PhoneticPipeline:
    def __init__(self, triggers):
        self.profiler = BlockProfiler(triggers)
        self.synonym_engine = SynonymEngine(triggers)
        self.scorer = ContextScorer()
        self.scorer.train_on_brown_corpus()

    def process_script(self, text):
        print("\n" + "="*50)
        print(f"ORIGINAL SCRIPT: '{text}'")
        print("="*50)

        flagged_words = self.profiler.flag_problematic_words(text)
        if not flagged_words:
            print("No problematic words detected. Script is safe to read.")
            return text
            
        print(f"Detected Block Triggers: {flagged_words}")
        
        safe_script = text
        
        for word in flagged_words:
            print(f"\nProcessing: '{word}'")
            candidates = self.synonym_engine.get_safe_synonyms(word)
            
            if not candidates:
                print(f" -> No safe synonyms found for '{word}'. Manual review required.")
                continue
                
            template = re.sub(rf'\b{word}\b', '___', safe_script, count=1, flags=re.IGNORECASE)
            best_replacement = self.scorer.score_candidates(template, candidates)
            
            if best_replacement:
                print(f" -> Selected Replacement: '{best_replacement}'")
                safe_script = re.sub(rf'\b{word}\b', best_replacement, safe_script, count=1, flags=re.IGNORECASE)
            else:
                print(f" -> Could not determine contextually safe replacement for '{word}'.")

        print("\n" + "="*50)
        print(f"SAFE SCRIPT: '{safe_script}'")
        print("="*50)
        
        return safe_script

if __name__ == "__main__":
    print("--- Phonetic: Full Neurosymbolic Pipeline Initialization ---")
    
    # 1. Run the Dynamic Assessment
    assessor = PangramAssessor()
    simulated_transcript = "p-pack my b-box with five dozen liquor j-jugs." # User blocked on p, b, j
    print(f"User reading pangram: '{simulated_transcript}'")
    
    dynamic_triggers = assessor.analyze_speech(simulated_transcript)
    print(f"User Profile Established! Triggers: {dynamic_triggers}")
    
    # 2. Initialize the pipeline with the dynamic triggers
    pipeline = PhoneticPipeline(triggers=dynamic_triggers)
    
    # 3. Process a new script that the user wants to read
    # Since 'p', 'b', and 'j' are triggers, "big", "presentation", and "job" should get flagged.
    test_script = "I need to design a big presentation for my new job."
    final_script = pipeline.process_script(test_script)