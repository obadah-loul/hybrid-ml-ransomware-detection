\# Hybrid ML Ransomware Detection



This repository contains the proof-of-concept implementation for the research paper:



\*\*A Hybrid Machine Learning Approach for Ransomware-Related Malicious Traffic Detection in Enterprise Networks\*\*



\## Overview



This project uses machine learning to detect ransomware-related malicious network traffic in enterprise environments.



The proposed approach combines:



\- Random Forest

\- XGBoost

\- Soft Voting Hybrid Ensemble



\## Datasets



The experiments were based on:



\- CICIDS2018

\- UNSW-NB15



The full datasets are not included in this repository because of size and storage limitations.



\## Repository Contents



\- `scripts/` — training, evaluation, and comparison scripts

\- `utils/` — preprocessing and evaluation utilities

\- `templates/` — Flask web interface templates

\- `results/` — confusion matrices, feature importance charts, and evaluation results

\- `app.py` — Flask proof-of-concept application

\- `requirements.txt` — Python dependencies



\## Proof of Concept



A Flask-based prototype was created to demonstrate how the trained model can be connected to a simple web interface for malicious traffic classification.



\## Results



The repository includes:



\- confusion matrices

\- model comparison charts

\- per-class reports

\- feature importance visualizations

\- summary statistics



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

