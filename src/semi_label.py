import pandas as pd
import os
import random

INPUT_PATH = "data/pokedex_entries.csv"
OUTPUT_PATH = "data/labeled_pokemon_entries.csv"

if not os.path.exists(INPUT_PATH):
    print(f"Error: Input file '{INPUT_PATH}' not found. Please make sure the file exists.")
    exit(1)

try:
    df = pd.read_csv(INPUT_PATH)
except pd.errors.EmptyDataError:
    print(f"Error: Input file '{INPUT_PATH}' is empty. Please add data with headers 'name, entry'.")
    exit(1)

if os.path.exists(OUTPUT_PATH):
    labeled_df = pd.read_csv(OUTPUT_PATH)
    # Track (name, entry) pairs that have been labeled
    labeled_pairs = set(zip(labeled_df["name"], labeled_df["entry"]))
else:
    labeled_df = pd.DataFrame(columns=["name", "entry", "sentiment"])
    labeled_pairs = set()

# Group entries by Pokémon name
grouped = df.groupby("name")

# Get a list of all Pokémon names and shuffle for random order
names = list(grouped.groups.keys())
random.shuffle(names)

for name in names:
    entries = grouped.get_group(name)["entry"].tolist()
    # Remove entries that have already been labeled for this Pokémon
    unlabeled_entries = [e for e in entries if (name, e) not in labeled_pairs]
    if not unlabeled_entries:
        continue
    # Randomly select one entry for labeling
    entry = random.choice(unlabeled_entries)

    print(f"\n{name.title()} : {entry}")
    label = input("Enter sentiment [P]Positive, [N]Negative, [E]Neutral, [S]Skip, [Q]Quit: ").strip().upper()

    if label == "Q":
        break
    elif label == "S":
        continue
    elif label not in ["P", "N", "E"]:
        print("Invalid input. Please enter P, N, E, S, or Q.")
        continue
    sentiment = "positive" if label == "P" else "negative" if label == "N" else "neutral"

    labeled_df = pd.concat([labeled_df, pd.DataFrame({
        "name": [name],
        "entry": [entry],
        "sentiment": [sentiment]
    })], ignore_index=True)

    labeled_df.to_csv(OUTPUT_PATH, index=False)
    labeled_pairs.add((name, entry))

print("\nLabeled entries saved to", OUTPUT_PATH)