import json
import torch
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

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
            f"Missing columns: {missing}",
            None
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
                "ticket_id": idx,
                "assigned_priority": df.iloc[idx]["priority_level"],
                "binary_judgment": "Mismatch Detected",
                "confidence": confidence_level,
                "confidence_score": round(float(mismatch_prob[idx]), 4),
                "mismatch_type": (
                    "Potential Under-Prioritization"
                    if df.iloc[idx]["priority_level"] in ["Low", "Medium"]
                    else "Potential Over-Prioritization"
                ),
                "evidence": (
                    f"Channel: {df.iloc[idx]['ticket_channel']}"
                ),
                "analysis": (
                    f"Assigned priority "
                    f"{df.iloc[idx]['priority_level']} "
                    f"appears inconsistent with "
                    f"the inferred ticket severity."
                )
            }
        )

    output_df = pd.DataFrame(results)
    total_tickets = len(df)
    
    flagged_tickets = len(output_df)

    flag_rate = (
        flagged_tickets / total_tickets * 100
        if total_tickets > 0
        else 0
    )

    avg_confidence = (
        output_df["confidence_score"].mean()
        if flagged_tickets > 0
        else 0
    )

    dashboard_summary = f"""
    Total Tickets: {total_tickets}
    Flagged Tickets: {flagged_tickets}
    Flag Rate: {flag_rate:.2f}%
    Average Confidence: {avg_confidence:.3f}
    """


    fig, ax = plt.subplots()

    ax.pie(
        [
            total_tickets - flagged_tickets,
            flagged_tickets
        ],
        labels=[
            "Consistent",
            "Flagged"
        ],
        autopct="%1.1f%%"
    )

    ax.set_title(
        "Priority Audit Distribution"
    )
    plt.tight_layout()

    return (
        output_df,
        dashboard_summary,
        fig
    )


demo = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(
        label="Upload Ticket CSV"
    ),
    outputs=[
        gr.Dataframe(
            label="Evidence Dossiers"
        ),
        gr.Textbox(
            label="Dashboard Summary"
        ),
        gr.Plot(
            label="Flag Distribution"
        )
    ],
    title="Priority Audit AI",
    description=(
        "Detects hidden priority mismatches "
        "in customer support tickets using "
        "signal-fusion based pseudo labeling "
        "and DistilBERT classification."
    )
)

demo.launch()