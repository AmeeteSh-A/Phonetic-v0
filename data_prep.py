import pandas as pd
import re

def get_synthetic_data():
    # Expanded dataset to balance the classes and provide more character-level variety
    data = {
        'raw_input': [
            'wnt wtr', 'nd hlp', 'bthrm pls', 'hngry f00d', 'thnk yu', 'pn in lg', 'n0 hrt', 'thsty wtr',
            'nd wtr', 'hlp me pls', 'tlt nw', 'need f00d', 'thnks a lt', 'hrt arm', 'im fne'
        ],
        'intent': [
            'request_water', 'request_help', 'request_bathroom', 'request_food', 'express_gratitude', 'report_pain', 'report_okay', 'request_water',
            'request_water', 'request_help', 'request_bathroom', 'request_food', 'express_gratitude', 'report_pain', 'report_okay'
        ],
        'symbolic_expansion': [
            'I would like some water.', 'I need help.', 'Please take me to the bathroom.', 'I am hungry and would like food.', 'Thank you.', 'I am experiencing pain in my leg.', 'I am not hurting.', 'I am thirsty and would like water.',
            'I need some water.', 'Please help me.', 'I need to use the toilet now.', 'I need some food.', 'Thanks a lot.', 'My arm hurts.', 'I am fine.'
        ]
    }
    return pd.DataFrame(data)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text

if __name__ == "__main__":
    print("--- Phonetic: Data Prep Initialization ---")
    df = get_synthetic_data()
    df['cleaned_input'] = df['raw_input'].apply(normalize_text)
    
    print("\nInitial Dataset Sample:")
    print(df[['raw_input', 'cleaned_input', 'intent']].head())