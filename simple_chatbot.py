import json
import random
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END, WORD
import re

# Load the intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

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

def get_response(user_input):
    user_input = user_input.lower()
    
    # Check for simple greetings first
    simple_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if user_input.strip() in simple_greetings or len(user_input.strip()) < 5:
        # Look for greeting intent
        for intent in intents['intents']:
            if intent['tag'] == "greeting":
                return format_response(random.choice(intent['responses']))
        
        # Fallback greeting if no greeting intent found
        return "Hello! How can I help you today?"
    
    # Improved matching algorithm
    best_match = None
    highest_score = 0
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern = pattern.lower()
            
            # Exact match
            if user_input == pattern:
                return format_response(random.choice(intent['responses']))
            
            # Contains match
            if pattern in user_input or any(word in user_input for word in pattern.split()):
                score = len(set(user_input.split()) & set(pattern.split())) / max(len(user_input.split()), len(pattern.split()))
                if score > highest_score:
                    highest_score = score
                    best_match = intent
    
    # If we found a decent match
    if best_match and highest_score > 0.2:
        return format_response(random.choice(best_match['responses']))
    
    # Check for keywords in each intent
    for intent in intents['intents']:
        tag = intent['tag'].lower()
        if tag in user_input or any(word in user_input for word in tag.split('_')):
            return format_response(random.choice(intent['responses']))
    
    # Default response if no match found
    return "I'm not sure how to respond to that. Could you rephrase? You can ask me about hospitals, pharmacies, blood pressure, or adverse drug reactions."

# Create the GUI
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        root.title("Medical Assistant Chatbot")
        root.geometry("500x600")
        root.resizable(width=False, height=False)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, wrap=WORD, bg='white', height=25, width=60)
        self.chat_display.config(state='disabled', font=("Arial", 10))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # Message entry area
        self.msg_entry = Entry(root, width=40, bg='white', font=("Arial", 10))
        self.msg_entry.grid(row=1, column=0, padx=10, pady=10)
        self.msg_entry.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = Button(root, text="Send", width=10, bg="#f9a602", command=self.send_message, font=("Arial", 10, "bold"))
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        
        # Welcome message
        self.update_chat_display("Bot", "Hello! I'm a medical assistant chatbot. How can I help you today?\n\nYou can ask me about:\n- Hospitals and pharmacies\n- Blood pressure tracking\n- Adverse drug reactions\n- General medical information")
    
    def send_message(self, event=None):
        # Get user input
        msg = self.msg_entry.get().strip()
        self.msg_entry.delete(0, END)
        
        if msg:
            # Display user message
            self.update_chat_display("You", msg)
            
            # Get and display bot response
            response = get_response(msg)
            self.update_chat_display("Bot", response)
    
    def update_chat_display(self, sender, message):
        self.chat_display.config(state='normal')
        if sender == "You":
            self.chat_display.insert(END, f"{sender}: ", "user")
            self.chat_display.insert(END, f"{message}\n\n", "user_msg")
        else:
            self.chat_display.insert(END, f"{sender}: ", "bot")
            self.chat_display.insert(END, f"{message}\n\n", "bot_msg")
        self.chat_display.tag_config("user", foreground="#0000FF", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("user_msg", foreground="#0000FF", font=("Arial", 10))
        self.chat_display.tag_config("bot", foreground="#008000", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("bot_msg", foreground="#008000", font=("Arial", 10))
        self.chat_display.config(state='disabled')
        self.chat_display.see(END)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
