import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os
import traceback
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

# Add error handling for model loading
try:
    from tensorflow.keras.models import load_model
    model = load_model('chatbot_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model = None

# Add error handling for file loading
try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    print("Data files loaded successfully!")
except Exception as e:
    print(f"Error loading data files: {e}")
    traceback.print_exc()
    # Default values in case of error
    intents = {"intents": [{"tag": "error", "patterns": [], "responses": ["Sorry, I'm having trouble. Please try again later."]}]}
    words = []
    classes = []

# Load CSV data
try:
    train_data = pd.read_csv('train_data_chatbot.csv')
    validation_data = pd.read_csv('validation_data_chatbot.csv')
    print(f"Loaded {len(train_data)} training QA pairs and {len(validation_data)} validation QA pairs")
    
    # Combine both datasets
    qa_data = pd.concat([train_data, validation_data], ignore_index=True)
    
    # Create TF-IDF vectorizer for question matching
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(qa_data['short_question'].values.astype('U'))
    
    print("CSV data loaded and processed successfully!")
except Exception as e:
    print(f"Error loading CSV data: {e}")
    traceback.print_exc()
    qa_data = None

def clean_up_sentence(sentence):
    try:
        # tokenize the pattern - splitting words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stemming every word - reducing to base form
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        print(f"Error in clean_up_sentence: {e}")
        return sentence.lower().split()  # Fallback to simple splitting

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=False):
    try:
        # tokenizing patterns
        sentence_words = clean_up_sentence(sentence)
        # bag of words - vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,word in enumerate(words):
                if word == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % word)
        return(np.array(bag))
    except Exception as e:
        print(f"Error in bag_of_words: {e}")
        return np.zeros(len(words))  # Return zeros in case of error

def predict_class(sentence):
    try:
        if not words or not classes or model is None:
            return [{"intent": "error", "probability": "1.0"}]
            
        # filter below threshold predictions
        p = bag_of_words(sentence, words, show_details=False)
        
        # Handle empty input
        if np.sum(p) == 0:
            print("No words matched in vocabulary")
            return []
            
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        
        # If no results above threshold, take the highest one
        if not results and len(res) > 0:
            best_idx = np.argmax(res)
            results = [[best_idx, res[best_idx]]]
            
        # sorting strength probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        
        print("Prediction results:", return_list)
        return return_list
    except Exception as e:
        print(f"Error in predict_class: {e}")
        traceback.print_exc()
        return [{"intent": "error", "probability": "1.0"}]

def find_best_match_from_csv(user_question, threshold=0.5):
    """Find the best matching question-answer pair from CSV data"""
    try:
        if qa_data is None:
            return None, 0
        
        # Transform user question
        user_vector = tfidf_vectorizer.transform([user_question])
        
        # Calculate similarity with all questions
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Find best match
        best_match_index = similarity_scores.argmax()
        best_match_score = similarity_scores[best_match_index]
        
        if best_match_score >= threshold:
            return qa_data.iloc[best_match_index], best_match_score
        return None, 0
    except Exception as e:
        print(f"Error in CSV matching: {e}")
        return None, 0

def format_response(response):
    """Format the response text with proper capitalization and punctuation"""
    if not response or not isinstance(response, str):
        return response
        
    # Add space after punctuation if missing
    response = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', response)
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response)
    
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

def getResponse(ints, intents_json, user_input):
    try:
        # Check for simple greetings first
        simple_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if user_input.lower().strip() in simple_greetings or len(user_input.strip()) < 5:
            # Look for greeting intent
            for intent in intents_json['intents']:
                if intent['tag'] == "greeting":
                    return format_response(random.choice(intent['responses']))
            
            # Fallback greeting if no greeting intent found
            return "Hello! How can I help you today?"
        
        # First try to find a match in the CSV data
        # Only for queries that are sufficiently long
        if len(user_input.split()) > 3:
            csv_match, score = find_best_match_from_csv(user_input)
            if csv_match is not None and score > 0.6:
                print(f"Found CSV match with score {score}: {csv_match['short_question']}")
                return format_response(csv_match['short_answer'])
        
        # If no good CSV match, proceed with intent-based response
        # Add error handling for empty predictions
        if not ints:
            return "I'm not sure how to respond to that. Could you try asking about hospitals, pharmacies, blood pressure, or adverse drug reactions?"
        
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        
        # First try exact tag match
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return format_response(result)
                
        # If no exact match, try partial match
        for i in list_of_intents:
            if tag in i['tag'] or i['tag'] in tag:
                result = random.choice(i['responses'])
                return format_response(result)
                
        # If still no match, return default
        return "I'm not sure how to respond to that. Could you try asking about hospitals, pharmacies, blood pressure, or adverse drug reactions?"
    except Exception as e:
        print(f"Error in getResponse: {e}")
        return "Sorry, I encountered an error. Please try again."

#Creating tkinter GUI
import tkinter
from tkinter import *

def send(event=None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n', "user")
        ChatBox.tag_configure("user", foreground="#0000FF", font=("Verdana", 12, "bold"))
    
        try:
            ints = predict_class(msg)
            res = getResponse(ints, intents, msg)
        except Exception as e:
            print(f"Error processing message: {e}")
            res = "Sorry, I encountered an error. Please try again."
        
        ChatBox.insert(END, "Bot: " + res + '\n\n', "bot")
        ChatBox.tag_configure("bot", foreground="#008000", font=("Verdana", 12))
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

# Initialize the GUI
root = Tk()
root.title("Medical Assistant Chatbot")
root.geometry("500x600")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="20", width="50", font=("Verdana", 12))
ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font=("Verdana", 12))
EntryBox.bind("<Return>", lambda event: send())

# Place all components on the screen
scrollbar.place(x=476,y=6, height=486)
ChatBox.place(x=6,y=6, height=486, width=470)
EntryBox.place(x=128, y=501, height=90, width=365)
SendButton.place(x=6, y=501, height=90)

# Add welcome message
ChatBox.config(state=NORMAL)
ChatBox.insert(END, "Bot: Hello! I'm a medical assistant chatbot. How can I help you today?\n\nYou can ask me about:\n- Hospitals and pharmacies\n- Blood pressure tracking\n- Adverse drug reactions\n- General medical information\n\n", "bot")
ChatBox.tag_configure("bot", foreground="#008000", font=("Verdana", 12))
ChatBox.config(state=DISABLED)

root.mainloop()
