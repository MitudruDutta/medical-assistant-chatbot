# Medical Assistant Chatbot

An intelligent medical assistant chatbot built with Python, TensorFlow, and natural language processing techniques.

## Features

- Multiple user interface options (simple, GUI, and advanced versions)
- Intent-based conversational AI using a trained neural network
- Medical knowledge from a curated dataset of healthcare Q&A
- Proper response formatting with correct grammar and punctuation
- Specialized handling for various medical topics including:
  - Hospital and pharmacy information
  - Blood pressure monitoring
  - Adverse drug reactions
  - General medical information

## Components

- `advanced_chatbot.py` - The full-featured chatbot with advanced NLP and GUI
- `gui_chatbot.py` - Intermediate version with graphical user interface
- `simple_chatbot.py` - A lightweight text-based chatbot
- `train_chatbot.py` - Script to train the neural network model
- `intents.json` - Intent definitions for conversational understanding
- `fix_csv_data.py` - Utility to format text data correctly

## Data

The chatbot uses two primary data sources:
- Intent-based responses from `intents.json`
- Medical Q&A dataset (full CSV files excluded from repository due to size)
- Sample data provided in `sample_data/` directory

### Full Dataset

The full training dataset contains over 50,000 medical Q&A pairs. Due to size limitations, these files are not included in the repository. To use the complete functionality:

1. Request access to the full dataset or create your own medical QA dataset with columns:
   - `short_question`: The medical question
   - `short_answer`: The corresponding medical answer

2. Place the CSV files in the root directory:
   - `train_data_chatbot.csv`
   - `validation_data_chatbot.csv`

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NLTK
- pandas
- scikit-learn
- tkinter (for GUI versions)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Usage

### Simple Chatbot:
```
python simple_chatbot.py
```

### GUI Chatbot:
```
python gui_chatbot.py
```

### Advanced Chatbot:
```
python advanced_chatbot.py
```

## Training the Model

To train the neural network model:
```
python train_chatbot.py
```

## License

[MIT License](LICENSE) 