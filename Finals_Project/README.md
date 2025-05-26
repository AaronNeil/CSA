# Pokémon Sentiment Analysis

This project builds a sentiment analysis pipeline for Pokémon Pokédex entries using modern NLP techniques and transformer models. It allows you to fetch entries, label them with sentiment, train a classifier, and analyze new entries.

---

## Project Structure

```
src/
├── analyze_sentiment.py     
├── fetch_entries.py          
├── semi_label.py             
├── train_model.py            
data/
├── pokedex_entries.csv       
├── labeled_pokemon_entries.csv 
models/
└── pokemon_sentiment_transformer/ 
```

---

## Workflow Overview

1. **Fetch Pokédex Entries**
   - Run `fetch_entries.py` to download and clean Pokédex entries from the PokeAPI.
   - Output: `data/pokedex_entries.csv`

2. **Label Entries**
   - Use `semi_label.py` to interactively assign sentiment (`positive`, `negative`, `neutral`) to entries.
   - Output: `data/labeled_pokemon_entries.csv`

3. **Train the Model**
   - Run `train_model.py` to preprocess text, encode labels, and fine-tune a transformer (DistilBERT) for sentiment classification.
   - Output: Model and artifacts saved in `models/pokemon_sentiment_transformer/`

4. **Analyze New Entries**
   - Use `analyze_sentiment.py` to predict the sentiment of any Pokédex entry using the trained model.

---

## Key Features

- **Modern NLP:** Uses Hugging Face Transformers for state-of-the-art text classification.
- **Interactive Labeling:** Semi-automated labeling tool for efficient dataset creation.
- **Metrics:** Reports accuracy, precision, recall, and F1-score.
- **Reusable Artifacts:** Saves model, tokenizer, and label encoder for future inference.

---

## How Each Script Works

### `fetch_entries.py`
- Fetches all Pokémon species from the PokeAPI.
- Cleans and saves English Pokédex entries to `data/pokedex_entries.csv`.

### `semi_label.py`
- Loads entries and lets you assign sentiment labels interactively.
- Appends new labels to `data/labeled_pokemon_entries.csv`.

### `train_model.py`
This script is responsible for training the sentiment analysis model. Here’s how it works, step by step:

1. **Imports and Setup**
   - Loads all required libraries for data handling, NLP preprocessing, model training, and evaluation.
   - Sets constants for the data path and model output directory.

2. **NLTK Resource Download**
   - Downloads necessary NLTK resources for tokenization and lemmatization.

3. **Text Preprocessing**
   - Splits each Pokédex entry into sentences.
   - Tokenizes each sentence into words.
   - Applies lemmatization and stemming to each token using NLTK.
   - Recombines tokens and sentences into a normalized string.

4. **Data Loading and Preparation**
   - Loads the labeled CSV file.
   - Encodes sentiment labels as integers using `LabelEncoder`.
   - Applies the preprocessing function to the `"entry"` column.

5. **Dataset Tokenization**
   - Converts the train and test DataFrames into Hugging Face `Dataset` objects.
   - Tokenizes the text entries for the transformer model.

6. **Metrics Computation**
   - Defines a function to compute accuracy, precision, recall, and F1-score using scikit-learn.

7. **Model Training**
   - Loads a pretrained transformer model (`distilbert-base-uncased`) for sequence classification.
   - Sets up training arguments (epochs, batch size, logging, etc.).
   - Initializes the Hugging Face `Trainer` with the model, datasets, and metrics.
   - Trains the model and evaluates it on the test set.

8. **Saving Artifacts**
   - Saves the trained model, tokenizer, and label encoder for future use in the `models/pokemon_sentiment_transformer/` directory.

---

### `analyze_sentiment.py`
This script is used for predicting the sentiment of new Pokédex entries using the trained model:

1. **Model and Data Loading**
   - Checks for the existence of the trained model directory and the Pokédex entries CSV.
   - Loads the trained transformer model, tokenizer, and label encoder.

2. **User Interaction**
   - Prompts the user to enter a Pokémon name.
   - Looks up the corresponding Pokédex entry (or entries) from the CSV file.

3. **Prediction**
   - Tokenizes the selected entry using the same tokenizer as during training.
   - Runs the entry through the trained model to get a sentiment prediction.
   - Decodes the predicted label back to the original sentiment string using the label encoder.

4. **Output**
   - Displays the Pokédex entry and the predicted sentiment to the user.
   - Allows repeated predictions until the user chooses to quit.

---

## Architecture Flow

![Architecture Flow](architecture_flow.png)

These scripts together allow you to preprocess and label data, train a robust sentiment classifier, and interactively analyze new Pokémon entries for sentiment.

---

## Requirements

- Python 3.7+ 
- `pandas`
- `scikit-learn`
- `nltk`
- `torch`
- `transformers`
- `datasets`
- `joblib`
- `tqdm`
- `requests`

Install all requirements with:

[Download Python](https://www.python.org/downloads/)

```sh
py -m pip install pandas scikit-learn nltk torch transformers[torch] datasets joblib tqdm requests
```

---

## Usage

```sh
cd CSA/Finals_Project
```

1. **Fetch entries:**
   ```sh
   python src/fetch_entries.py
   ```

2. **Label entries(optional):**
   ```sh
   python src/semi_label.py
   ```

3. **Train the model:**
   ```sh
   python src/train_model.py
   ```

4. **Analyze sentiment:**
   ```sh
   python src/analyze_sentiment.py
   ```

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [PokeAPI](https://pokeapi.co/)