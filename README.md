# Attention-based Seq2Seq for Time Series Forecasting

This project explores **multi-step time series forecasting** using:

- **ARIMA** (baseline)
- **Seq2Seq LSTM** (vanilla)
- **Seq2Seq with Bahdanau Attention** (improved model)

Walk-forward validation is used for realistic evaluation of the models.

---

## 📂 Repository Structure

attention-seq2seq-time-series/
│
├── code/ # Python scripts
├── data/ # Data folders (placeholders only)
├── images/ # Optional screenshots
├── README.md # This file
└── .gitignore


- **code/** contains all scripts:
  - `01_data_generation.py` → Synthetic multivariate dataset
  - `02_preprocessing.py` → Train/val/test split & scaling
  - `03_baseline_arima.py` → ARIMA baseline model
  - `04_baseline_seq2seq.py` → Vanilla Seq2Seq model
  - `05_attention_seq2seq.py` → Seq2Seq with attention
  - `06_walk_forward_validation.py` → Walk-forward evaluation
  - `07_evaluation.py` → Metrics calculation (RMSE, MAE, MAPE)
  - `08_visualization.py` → Plots & attention heatmaps

- **data/** contains:
  - `raw/` → original data (empty placeholder `.gitkeep`)
  - `processed/` → processed dataset (empty placeholder `.gitkeep`)

---

## 📝 Project Overview

### Objective
To compare **vanilla Seq2Seq** vs **Seq2Seq with Attention** for time series forecasting, and validate if attention improves accuracy and interpretability.

### Dataset
- Synthetic multivariate time series of **daily energy demand**
- Features:
  1. Temperature (annual sinusoidal + noise)
  2. Humidity (weekly cycle)
  3. Industrial load (volatility + trend)
- Target: `energy_demand` (weighted mixture of features)
- Split into **train, validation, test** sets in temporal order

### Preprocessing
- Scaling features to stabilize training
- Sliding windows for sequence modeling
- Decoder inputs prepared for **teacher forcing**

---

## 🧠 Models Implemented

1. **ARIMA** → Baseline linear model  
2. **Seq2Seq LSTM (Vanilla)** → Encoder-decoder predicts multi-step  
3. **Seq2Seq with Bahdanau Attention** → Focuses on important time steps, improves accuracy, adds interpretability

---

## 🛠 Hyperparameters

- Hidden units: 64
- Attention units: 32 (tuned)
- Learning rate: 0.0005 (tuned)
- Batch size: 32
- Epochs: 10
- Output steps: 14
- Input steps: 60

---

## 📊 Walk-forward Validation

- Stepwise multi-step forecasting
- Evaluates **accumulated errors** over time
- More realistic than static test evaluation

### Metrics Collected

| Model                  | RMSE    | MAE    | MAPE   |
|------------------------|--------|--------|--------|
| Seq2Seq (Baseline)     | 0.0987 | 0.0743 | 9.82%  |
| Seq2Seq + Attention    | 0.0842 | 0.0615 | 7.34%  |

---

## 📈 Visualizations

- Forecast vs True plots
- Attention heatmaps highlighting important encoder time steps

![Forecast Example](images/forecast_sample.png)
![Attention Heatmap](images/attention_heatmap.png)

---

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/abinithi-ka/attention-seq2seq-time-series.git
cd attention-seq2seq-time-series/code
Install requirements:

pip install -r requirements.txt
Run scripts in order:

python 01_data_generation.py
python 02_preprocessing.py
python 03_baseline_arima.py
python 04_baseline_seq2seq.py
python 05_attention_seq2seq.py
python 06_walk_forward_validation.py
python 07_evaluation.py
python 08_visualization.py
Ensure data/raw and data/processed exist (even as empty placeholders)

```
🔑 Key Takeaways

Attention improves forecast accuracy in multi-step time series

Walk-forward validation provides realistic error estimation

Attention weights provide interpretability

📄 References

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning

