import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")


DATA_PATH = "CSA/Finals_Project/data/labeled_pokemon_entries.csv"
MODEL_OUT = "CSA/Finals_Project/models/pokemon_sentiment_transformer"

def download_nltk_resources():
    nltk.download('punkt')

def preprocess_text(text):
    sentences = sent_tokenize(text, language='english')
    processed_sentences = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        processed_sentences.append(' '.join(tokens))
    return ' '.join(processed_sentences)

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Labeled data file '{DATA_PATH}' not found.")
        exit(1)
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        print("No labeled data available for training.")
        exit(1)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    df["entry"] = df["entry"].astype(str).apply(preprocess_text)
    return df, le

def tokenize_datasets(train_df, test_df, tokenizer):
    train_ds = Dataset.from_pandas(train_df[["entry", "label"]])
    test_ds = Dataset.from_pandas(test_df[["entry", "label"]])
    def preprocess(batch):
        return tokenizer(batch["entry"], truncation=True, padding="max_length", max_length=128)
    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_ds, test_ds

def compute_metrics_factory(le):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        all_labels = list(range(len(le.classes_)))
        report = classification_report(
            labels, preds,
            labels=all_labels,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )
        return {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        }
    return compute_metrics

def main():
    download_nltk_resources()
    df, le = load_and_prepare_data()
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, test_ds = tokenize_datasets(train_df, test_df, tokenizer)

    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=MODEL_OUT,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"{MODEL_OUT}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics_factory(le),
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    joblib.dump(le, os.path.join(MODEL_OUT, "label_encoder.pkl"))
    print(f"Model and tokenizer saved to '{MODEL_OUT}'")

if __name__ == "__main__":
    main()