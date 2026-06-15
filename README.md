# Priority Audit AI

## Overview

Priority Audit AI is an AI-powered auditing system that detects inconsistencies between the priority assigned to a customer support ticket and the severity inferred from its content.

Rather than directly predicting ticket priority, the system acts as an independent auditor that flags potentially misclassified tickets, helping support teams uncover hidden crises, reduce false alarms, and improve triaging decisions.

**Hosted Application:**
https://huggingface.co/spaces/iitMaan/Priority-Audit-AI

---

## Problem Statement

Customer support tickets are often assigned priorities manually. This process can lead to:

* Critical incidents being assigned low priority.
* Routine issues being escalated unnecessarily.
* Inconsistent prioritization across support channels.
* Delayed response to business-critical issues.

Priority Audit AI automatically audits assigned priorities and highlights tickets that may require review.

---

## Methodology

### Signal Fusion Based Pseudo-Labeling

Since ground-truth severity labels were unavailable, a pseudo-labeling framework was developed using two independent severity signals.

### Signal 1: NLP Severity Extraction

Ticket descriptions are analyzed using severity-oriented keyword rules.

Examples:

* Critical: outage, breach, fraud, security, fatal
* High: urgent, escalate, crash, broken
* Medium: refund, delay, issue, waiting

### Signal 2: Resolution Time Analysis

Historical resolution behavior is used as a secondary severity indicator.

Tickets resolved significantly faster than the category median are treated as more severe, while unusually slow resolutions indicate lower urgency.

### Severity Fusion

The two signals are combined using weighted fusion:

* NLP Signal Weight: 80%
* Resolution Signal Weight: 20%

The resulting pseudo-severity serves as the reference severity used during training.

---

## Mismatch Detection Framework

Assigned Priority

↓

Pseudo Severity

↓

Binary Mismatch Label

The task is formulated as a binary classification problem:

* Consistent
* Mismatch Detected

This approach allows the model to act as an auditing layer rather than a priority prediction engine.

---

## Model Architecture

### DistilBERT Binary Classifier

Input Features:

* Assigned Priority
* Ticket Channel
* Ticket Subject
* Ticket Description

Model:

* DistilBERT Base Uncased
* Binary Classification Head

Output:

* Mismatch Probability
* Binary Judgment
* Confidence Score

---

## Evidence Dossier Generation

For every flagged ticket, the system generates:

* Ticket ID
* Assigned Priority
* Binary Judgment
* Confidence Level
* Confidence Score
* Mismatch Type
* Supporting Evidence
* Constraint Analysis

This ensures transparent and explainable auditing decisions.

---

## Dashboard Features

The hosted application provides:

* CSV Upload Interface
* Automated Ticket Auditing
* Evidence Dossier Generation
* Flagged Ticket Summary
* Confidence Analysis
* Priority Mismatch Distribution Chart
* Audit Statistics Dashboard

---

## Repository Structure

```text
Priority-Audit-AI/
│
├── notebook.ipynb
├── train_pipeline.py
├── predict.py
├── app.py
├── requirements.txt
├── sample_tickets.csv
│
└── model/
```

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

---

## Example Input Format

Required CSV columns:

```text
priority_level
ticket_channel
ticket_subject
ticket_description
```

Example:

```csv
priority_level,ticket_channel,ticket_subject,ticket_description
Low,Email,Server Outage,Production servers are down globally
Medium,Chat,Refund Request,Customer requesting refund
High,Phone,Security Breach,Sensitive customer information may have been exposed
```

---

## Output

The system produces:

* Evidence Dossiers
* Mismatch Probability Scores
* Confidence Levels
* Dashboard Metrics
* Priority Distribution Visualization

---

## Future Improvements

* Multi-class severity prediction
* Agent-level audit analytics
* Explainable AI signal attribution
* Real-time ticket monitoring
* Advanced severity calibration

---

## Authors

Developed as part of the SIA Hackathon submission.

Priority Audit AI demonstrates how pseudo-labeling, signal fusion, and transformer-based NLP can be combined to build an effective auditing framework for customer support ticket prioritization.
