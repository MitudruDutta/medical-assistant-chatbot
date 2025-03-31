import pandas as pd
import numpy as np
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
import tkinter as tk
from tkinter import scrolledtext, Text, Button, END, WORD, Scrollbar, Frame, Label, NORMAL, DISABLED
import traceback
import re
import os

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class AdvancedMedicalChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Load intent data
        try:
            self.intents = json.loads(open('intents.json').read())
            print("Intents loaded successfully")
        except Exception as e:
            print(f"Error loading intents: {e}")
            self.intents = {"intents": []}
        
        # Load neural network model
        try:
            self.words = pickle.load(open('words.pkl', 'rb'))
            self.classes = pickle.load(open('classes.pkl', 'rb'))
            self.model = load_model('chatbot_model.h5')
            print("Neural network model loaded successfully")
            self.nn_model_available = True
        except Exception as e:
            print(f"Error loading neural network model: {e}")
            self.nn_model_available = False
        
        # Load CSV data
        try:
            train_data = pd.read_csv('train_data_chatbot.csv')
            validation_data = pd.read_csv('validation_data_chatbot.csv')
            self.qa_data = pd.concat([train_data, validation_data], ignore_index=True)
            
            # Filter out rows with NaN or empty answers
            self.qa_data = self.qa_data.dropna(subset=['short_answer'])
            self.qa_data = self.qa_data[self.qa_data['short_answer'].str.strip() != '']
            
            # Create TF-IDF vectorizer for questions
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english', 
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                max_features=5000,
                min_df=2
            )
            
            # Process questions to handle text properly
            processed_questions = self.qa_data['short_question'].fillna('').astype(str)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_questions)
            print(f"CSV data loaded successfully: {len(self.qa_data)} QA pairs")
            self.csv_data_available = True
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            traceback.print_exc()
            self.csv_data_available = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_up_sentence(self, sentence):
        """Tokenize and lemmatize a sentence"""
        try:
            # Tokenize words
            sentence_words = word_tokenize(sentence)
            # Lemmatize each word
            sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            return sentence_words
        except Exception as e:
            print(f"Error in clean_up_sentence: {e}")
            return sentence.lower().split()
    
    def bag_of_words(self, sentence):
        """Convert sentence to bag of words array"""
        try:
            sentence_words = self.clean_up_sentence(sentence)
            bag = [0] * len(self.words)
            
            for s in sentence_words:
                for i, word in enumerate(self.words):
                    if word == s:
                        bag[i] = 1
            
            return np.array(bag)
        except Exception as e:
            print(f"Error in bag_of_words: {e}")
            return np.zeros(len(self.words))
    
    def predict_intent(self, sentence):
        """Predict intent class based on sentence using neural network"""
        try:
            if not self.nn_model_available:
                return []
                
            # Generate bag of words
            bow = self.bag_of_words(sentence)
            
            # Get prediction from neural network
            res = self.model.predict(np.array([bow]))[0]
            
            # Filter results above threshold
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            
            # If no results above threshold, take the highest one
            if not results and len(res) > 0:
                best_idx = np.argmax(res)
                results = [[best_idx, res[best_idx]]]
                
            # Sort by probability
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            result_list = []
            for r in results:
                # Skip adding adverse_drug intent unless it has a very high confidence score
                if self.classes[r[0]] == "adverse_drug" and r[1] < 0.8:
                    continue
                    
                result_list.append({
                    "intent": self.classes[r[0]],
                    "probability": float(r[1])
                })
            
            return result_list
        except Exception as e:
            print(f"Error in predict_intent: {e}")
            traceback.print_exc()
            return []
    
    def find_csv_match(self, query, threshold=0.55):
        """Find best matching question-answer pair from CSV data"""
        try:
            if not self.csv_data_available:
                return None, 0
            
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate similarity with all questions
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Find top matches
            top_indices = similarity_scores.argsort()[-5:][::-1]
            
            # Check if best match exceeds threshold
            best_match_index = top_indices[0]
            best_match_score = similarity_scores[best_match_index]
            
            if best_match_score >= threshold:
                # Return the match and its score
                return self.qa_data.iloc[best_match_index], best_match_score
            
            # No good match found
            return None, 0
        except Exception as e:
            print(f"Error in find_csv_match: {e}")
            return None, 0
    
    def keyword_match(self, query):
        """Match query with intents based on keywords"""
        try:
            query_words = set(self.clean_up_sentence(query))
            best_match = None
            highest_score = 0
            
            for intent in self.intents['intents']:
                # Skip adverse_drug intent unless explicitly mentioned
                if intent['tag'] == "adverse_drug" and not any(word in query.lower() for word in 
                    ["adverse", "drug reaction", "side effect", "drug", "medication reaction"]):
                    continue
                    
                # Check each pattern in the intent
                for pattern in intent['patterns']:
                    pattern_words = set(self.clean_up_sentence(pattern))
                    
                    # Calculate word overlap score
                    if pattern_words:
                        overlap = len(query_words.intersection(pattern_words))
                        score = overlap / max(len(query_words), len(pattern_words))
                        
                        if score > highest_score:
                            highest_score = score
                            best_match = intent
            
            if highest_score > 0.3:  # Threshold for keyword matching
                return best_match, highest_score
            
            return None, 0
        except Exception as e:
            print(f"Error in keyword_match: {e}")
            return None, 0
    
    def format_response(self, response):
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
    
    def get_response(self, query):
        """Generate a response based on the user's query"""
        try:
            # Check for simple greetings first
            simple_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
            if query.lower().strip() in simple_greetings or len(query.strip()) < 5:
                # Look for greeting intent
                for intent in self.intents['intents']:
                    if intent['tag'] == "greeting":
                        return self.format_response(random.choice(intent['responses']))
                
                # Fallback greeting if no greeting intent found
                return "Hello! How can I help you today?"
            
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Check if query explicitly mentions adverse drug reactions
            is_adverse_drug_query = any(term in processed_query for term in 
                ["adverse drug", "drug reaction", "side effect", "adverse reaction", "drug allergy"])
            
            # Strategy 1: Try CSV match first (for specific medical questions)
            # Only for queries that are sufficiently long
            if len(processed_query.split()) > 3:
                csv_match, csv_score = self.find_csv_match(processed_query)
                if csv_match is not None and csv_score > 0.6:
                    print(f"CSV match (score={csv_score:.2f}): {csv_match['short_question']}")
                    return self.format_response(csv_match['short_answer'])
            
            # Strategy 2: Try neural network intent prediction
            intent_predictions = self.predict_intent(processed_query)
            if intent_predictions and intent_predictions[0]['probability'] > 0.7:
                intent_tag = intent_predictions[0]['intent']
                
                # Additional check for adverse_drug intent
                if intent_tag == "adverse_drug" and not is_adverse_drug_query:
                    # Skip this intent if not explicitly asked about adverse drug reactions
                    print(f"Skipping adverse_drug intent despite high score as query doesn't explicitly mention it")
                else:
                    print(f"Intent match (score={intent_predictions[0]['probability']:.2f}): {intent_tag}")
                    
                    # Find matching intent
                    for intent in self.intents['intents']:
                        if intent['tag'] == intent_tag:
                            # For adverse_drug intent, provide a more specific response
                            if intent_tag == "adverse_drug":
                                adverse_responses = [
                                    "I can provide information about adverse drug reactions. What specific medication are you concerned about?",
                                    "To check for adverse drug reactions, I'll need more information about the medication you're taking.",
                                    "I can help with adverse drug reaction information. Could you specify which medication you'd like to know about?"
                                ]
                                return random.choice(adverse_responses)
                            else:
                                return self.format_response(random.choice(intent['responses']))
            
            # Strategy 3: Try keyword matching as fallback
            keyword_intent, keyword_score = self.keyword_match(processed_query)
            if keyword_intent is not None and keyword_score > 0.3:
                print(f"Keyword match (score={keyword_score:.2f}): {keyword_intent['tag']}")
                
                # For adverse_drug intent, provide a more specific response
                if keyword_intent['tag'] == "adverse_drug":
                    adverse_responses = [
                        "I can provide information about adverse drug reactions. What specific medication are you concerned about?",
                        "To check for adverse drug reactions, I'll need more information about the medication you're taking.",
                        "I can help with adverse drug reaction information. Could you specify which medication you'd like to know about?"
                    ]
                    return random.choice(adverse_responses)
                else:
                    return self.format_response(random.choice(keyword_intent['responses']))
            
            # Strategy 4: If we have a lower confidence CSV match, use it as last resort
            if csv_match is not None and csv_score > 0.5:
                print(f"Lower confidence CSV match (score={csv_score:.2f}): {csv_match['short_question']}")
                return self.format_response(csv_match['short_answer'])
            
            # Default response when no match is found
            default_responses = [
                "I'm not sure I understand. Could you rephrase your question?",
                "I don't have that information. Could you ask something about medical topics, hospitals, or medications?",
                "I'm still learning about that topic. Could you ask about blood pressure, hospitals, or adverse drug reactions?",
                "I don't have enough information to answer that question properly. Could you provide more details?"
            ]
            return random.choice(default_responses)
        
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return "I'm experiencing a technical issue. Could you try again?"

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.chatbot = AdvancedMedicalChatbot()
        
        # Configure root window
        root.title("Advanced Medical Assistant")
        root.geometry("600x700")
        root.configure(bg="#f0f0f0")
        root.resizable(True, True)
        
        # Create main frame
        main_frame = Frame(root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title label
        title_label = Label(
            main_frame, 
            text="Advanced Medical Assistant", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(pady=10)
        
        # Create chat display area
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=WORD,
            bg="white",
            font=("Arial", 12),
            width=50,
            height=20
        )
        self.chat_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_display.config(state=DISABLED)
        
        # Create input frame
        input_frame = Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Create entry for user input
        self.user_input = Text(
            input_frame,
            height=3,
            width=40,
            font=("Arial", 12),
            bg="white",
            bd=1
        )
        self.user_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.on_enter_pressed)
        
        # Create send button
        self.send_button = Button(
            input_frame,
            text="Send",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            relief="flat",
            command=self.send_message
        )
        self.send_button.pack(side="right", padx=(5, 0))
        
        # Add welcome message
        self.add_message("Bot", "Hello! I'm an Advanced Medical Assistant. How can I help you today?\n\nYou can ask me about:\n- Specific medical questions and treatments\n- Hospital and pharmacy information\n- Blood pressure monitoring\n- Adverse drug reactions\n- And more!")
        
        # Set focus to input field
        self.user_input.focus_set()
    
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=NORMAL)
        
        # Format based on sender
        if sender == "You":
            self.chat_display.insert(END, f"{sender}: ", "user_name")
            self.chat_display.insert(END, f"{message}\n\n", "user_msg")
        else:
            self.chat_display.insert(END, f"{sender}: ", "bot_name")
            self.chat_display.insert(END, f"{message}\n\n", "bot_msg")
        
        # Configure tags
        self.chat_display.tag_config("user_name", foreground="#3498db", font=("Arial", 12, "bold"))
        self.chat_display.tag_config("user_msg", foreground="#2c3e50", font=("Arial", 12))
        self.chat_display.tag_config("bot_name", foreground="#27ae60", font=("Arial", 12, "bold"))
        self.chat_display.tag_config("bot_msg", foreground="#2c3e50", font=("Arial", 12))
        
        # Disable and scroll to end
        self.chat_display.config(state=DISABLED)
        self.chat_display.see(END)
    
    def on_enter_pressed(self, event):
        """Handle Enter key press in input field"""
        # Don't add a newline
        self.send_message()
        return "break"  # Prevents default behavior (newline)
    
    def send_message(self):
        """Process and send the user message"""
        # Get user input
        msg = self.user_input.get("1.0", "end-1c").strip()
        
        # Clear input field
        self.user_input.delete("1.0", END)
        
        if msg:
            # Display user message
            self.add_message("You", msg)
            
            # Generate and display response
            response = self.chatbot.get_response(msg)
            self.add_message("Bot", response)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
