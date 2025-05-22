import requests
import pandas as pd
import os
import time
from tqdm import tqdm

DATA_PATH = "data/pokedex_entries.csv"
API_URL = "https://pokeapi.co/api/v2/pokemon-species/"

def clean_entry_text(text):
    # Clean and normalize Pokédex entry text
    if not text:
        return ""
    text = text.replace('\f', ' ')
    text = text.replace('\u00ad', '')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('"', '')
    text = ' '.join(text.split())
    return text

def save_progress(entries):
    # Save current progress to CSV.
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(entries).to_csv(DATA_PATH, index=False)

def get_english_entries(entry_url):
    # Fetch all English Pokédex entries for a Pokémon.
    try:
        entry_response = requests.get(entry_url)
        entry_response.raise_for_status()
        entry_data = entry_response.json()
        english_entries = []
        for entry in entry_data.get("flavor_text_entries", []):
            if entry.get("language", {}).get("name") == "en":
                cleaned = clean_entry_text(entry.get("flavor_text"))
                if cleaned and cleaned not in english_entries:
                    english_entries.append(cleaned)
        return english_entries
    except requests.RequestException as e:
        print(f"Error fetching entry: {e}")
        if hasattr(e.response, 'status_code') and e.response.status_code == 429:
            print("Rate limited. Sleeping for 60 seconds...")
            time.sleep(60)
    except Exception as e:
        print(f"Error processing entry: {e}")
    return []

def process_pokemon(pokemon, seen, entries):
    # Process a single Pokémon and update entries if successful.
    name = pokemon.get("name")
    entry_url = pokemon.get("url")
    if not name or not entry_url or name in seen:
        return False
    entry_texts = get_english_entries(entry_url)
    if not entry_texts:
        return False
    for entry_text in entry_texts:
        entries.append({"name": name, "entry": entry_text})
    seen.add(name)
    return True

def fetch_pokedex_entries():
    entries = []
    seen = set()

    # Resume support: load existing entries if file exists and is not empty
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        df_existing = pd.read_csv(DATA_PATH)
        seen = set(df_existing["name"])
        entries = df_existing.to_dict("records")

    # Get total count for progress bar
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
    total_count = data.get("count", 0)
    pbar = tqdm(total=total_count, desc="Processing all Pokemon", initial=len(seen))

    url = API_URL
    while url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from PokeAPI: {e}")
            print("Sleeping for 60 seconds before retrying...")
            time.sleep(60)
            continue

        for pokemon in data.get("results", []):
            success = process_pokemon(pokemon, seen, entries)
            if success and len(entries) % 10 == 0:
                save_progress(entries)
            time.sleep(0.1)
            pbar.update(1)
        url = data.get("next")
    pbar.close()
    if entries:
        save_progress(entries)
        print(f"Pokedex entries fetched and saved to '{DATA_PATH}'.")
    else:
        print("No entries were fetched.")

if __name__ == "__main__":
    fetch_pokedex_entries()