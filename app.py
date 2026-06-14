import json
import torch
import pandas as pd
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_DIR = "model"

# Load model once when the Space starts
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


def build_model_input(row):
    return (
        f"Priority={row['priority_level']} | "
        f"Channel={row['ticket_channel']} | "
        f"Subject={row['ticket_subject']} | "
        f"Description={row['ticket_description']}"
    )


def analyze_csv(file):

    df = pd.read_csv(file.name)

    required_columns = [
        "priority_level",
        "ticket_channel",
        "ticket_subject",
        "ticket_description"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        return (
            pd.DataFrame(),
            f"Missing columns: {missing}"
        )

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

    mismatch_prob = probs[:, 1]

    predictions = (
        mismatch_prob >= 0.5
    ).astype(int)

    results = []

    for idx, pred in enumerate(predictions):

        if pred != 1:
            continue

        confidence = float(
            max(probs[idx])
        )

        if confidence >= 0.85:
            confidence_level = "High"
        elif confidence >= 0.65:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        results.append(
            {
                "row_index": idx,
                "assigned_priority":
                    df.iloc[idx]["priority_level"],
                "mismatch_probability":
                    round(
                        float(mismatch_prob[idx]),
                        4
                    ),
                "confidence":
                    confidence_level
            }
        )

    output_df = pd.DataFrame(results)

    summary = (
        f"Processed {len(df)} tickets.\n"
        f"Detected {len(output_df)} "
        f"potential priority mismatches."
    )

    return output_df, summary


demo = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(
        label="Upload Ticket CSV"
    ),
    outputs=[
        gr.Dataframe(
            label="Detected Mismatches"
        ),
        gr.Textbox(
            label="Summary"
        )
    ],
    title="Customer Support Ticket Auditor",
    description=(
        "Upload a CSV of support tickets. "
        "The model identifies tickets whose "
        "assigned priority may not match "
        "their inferred severity."
    )
)

demo.launch()