import pandas as pd
import re
import string

def capitalize_sentences(text):
    """Capitalize the first letter of each sentence and fix punctuation"""
    if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
        return text
    
    # Add space after punctuation if missing
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Capitalize first letter of each sentence and ensure proper ending
    formatted_sentences = []
    for sentence in sentences:
        if sentence.strip():
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Ensure sentence ends with proper punctuation
            if not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
                
            formatted_sentences.append(sentence)
    
    return ' '.join(formatted_sentences)

def fix_csv_formatting():
    print("Starting CSV formatting...")
    
    try:
        # Load the CSV files
        train_data = pd.read_csv('train_data_chatbot.csv')
        validation_data = pd.read_csv('validation_data_chatbot.csv')
        
        print(f"Loaded training data: {len(train_data)} rows")
        print(f"Loaded validation data: {len(validation_data)} rows")
        
        # Fix formatting in training data
        print("Formatting training data...")
        train_data['short_question'] = train_data['short_question'].apply(capitalize_sentences)
        train_data['short_answer'] = train_data['short_answer'].apply(capitalize_sentences)
        
        # Fix formatting in validation data
        print("Formatting validation data...")
        validation_data['short_question'] = validation_data['short_question'].apply(capitalize_sentences)
        validation_data['short_answer'] = validation_data['short_answer'].apply(capitalize_sentences)
        
        # Save back to the original CSV files
        print("Saving formatted files...")
        train_data.to_csv('train_data_chatbot.csv', index=False)
        validation_data.to_csv('validation_data_chatbot.csv', index=False)
        
        print("Formatting complete!")
        print(f"Saved formatted training data: {len(train_data)} rows")
        print(f"Saved formatted validation data: {len(validation_data)} rows")
        
    except Exception as e:
        print(f"Error formatting CSV files: {e}")

if __name__ == "__main__":
    fix_csv_formatting() 