# Analitica_Final_Project

This project implements data ingestion, ML modeling, and explainability (XAI) for financial data, along with a Streamlit dashboard for visualization.

## Project Structure
```
Analitica_Final_Project/
│
├── app.py
├── utils.py
├── ml_models.py
├── xai.py
│
├── data/
├── models/
├── imgs/
├── pages/
│   ├── 1.Stock_Overview.py
│   ├── 2.Stock_Prediction.py
│   ├── 3.Trading_Simulation.py
│   ├── 4.AI_Explainability.py
├── requirements.txt
└── README.md
```

## Create and activate the environment
```
python3 -m venv .venv

.\.venv\Scripts\activate
```
## Install dependencies
```
pip install -r requirements.txt
```

## Run preprocessing & model scripts
```
python -m utils
python -m ml_models
python  -m xai
```

## Launch streamlit app
```
python -m streamlit run app.py
```