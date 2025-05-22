import os
import sys
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

MODEL_DIR = "CSA/Finals_Project/models/pokemon_sentiment_transformer"
DATA_PATH = "CSA/Finals_Project/data/pokedex_entries.csv"

def main():
    # Check if model and data exist
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found. Train the model first.")
        sys.exit(1)
    if not os.path.exists(DATA_PATH):
        print(f"Error: Pokédex entries file '{DATA_PATH}' not found.")
        sys.exit(1)

    # Load model, tokenizer, and label encoder
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Load pokedex entries
    df = pd.read_csv(DATA_PATH)
    df["name"] = df["name"].str.lower()

    print("Type a Pokémon name to analyze its Pokédex entry sentiment (or type 'quit' to exit):")
    while True:
        name = input("\nPokémon name: ").strip().lower()
        if name == "quit":
            break

        rows = df[df["name"] == name]
        if rows.empty:
            print("Pokémon not found in Pokédex entries.")
            continue

        # Randomly select one entry if multiple exist
        entry = rows.sample(1)["entry"].values[0]

        # Tokenize and predict
        inputs = tokenizer(entry, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label_id = outputs.logits.argmax(dim=1).item()
            sentiment = le.inverse_transform([pred_label_id])[0]

        print(f"\nPokédex entry: {entry}")
        print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()