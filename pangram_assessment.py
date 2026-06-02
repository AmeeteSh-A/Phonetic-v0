import re

class PangramAssessor:
    def __init__(self):
        # A classic phonetically dense pangram to test major consonants
        self.target_script = "Pack my box with five dozen liquor jugs."

    def analyze_speech(self, transcribed_text):
        print("Analyzing speech pattern for phonetic blocks...")
        triggers = set()
        text = transcribed_text.lower()
        
        # Regex to find stutters (e.g., "p-pack" or "st-start")
        # Looks for 1-2 letters followed by a hyphen, attached to a word
        stutter_pattern = r'\b([a-z]{1,2})-([a-z]+)\b'
        
        words = text.split()
        for word in words:
            match = re.match(stutter_pattern, word)
            if match:
                trigger_sound = match.group(1)
                word_body = match.group(2)
                
                # Verify it's a true block (e.g., 'p' in 'p-pack')
                if word_body.startswith(trigger_sound):
                    triggers.add(trigger_sound)

        return list(triggers)

if __name__ == "__main__":
    print("--- Phonetic: Dynamic Profile Assessment ---")
    assessor = PangramAssessor()
    print(f"Target Assessment Script: '{assessor.target_script}'")
    
    # Simulating a user struggling with 'p', 'b', and 'd' sounds
    simulated_transcript = "p-pack my b-box with five d-dozen liquor jugs."
    print(f"\nSimulated Audio Transcript: '{simulated_transcript}'")
    
    dynamic_triggers = assessor.analyze_speech(simulated_transcript)
    print(f"\n=> Dynamically Generated Triggers: {dynamic_triggers}")