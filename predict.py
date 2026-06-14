import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_DIR = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()


def build_model_input(row):
    return (
        f"Priority={row['priority_level']} | "
        f"Channel={row['ticket_channel']} | "
        f"Subject={row['ticket_subject']} | "
        f"Description={row['ticket_description']}"
    )


def predict(csv_path):

    df = pd.read_csv(csv_path)

    texts = df.apply(
        build_model_input,
        axis=1
    ).tolist()

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=-1
    ).cpu().numpy()

    df["mismatch_probability"] = probs[:, 1]
    df["prediction"] = (
        probs[:, 1] >= 0.5
    ).astype(int)

    return df


if __name__ == "__main__":

    results = predict(
        "sample_tickets.csv"
    )

    results.to_csv(
        "predictions.csv",
        index=False
    )

    print("Predictions saved.")