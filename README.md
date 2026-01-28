# Attention-based Seq2Seq for Time Series Forecasting

This project explores **multi-step time series forecasting** using:

- **ARIMA** (baseline)
- **Seq2Seq LSTM** (vanilla)
- **Seq2Seq with Bahdanau Attention** (improved model)

Walk-forward validation is used for realistic evaluation of the models.

---
## 📂 Repository Structure
```
attention-seq2seq-time-series/
├── code/ # Python scripts
├── data/ # Data folders (placeholders only)
├── images/ # Screenshots of plots
├── README.md # Project explanation & results
└── .gitignore

```

- **code/** contains all scripts:
  - `01_data_generation.py` → Synthetic dataset
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
Compare **vanilla Seq2Seq** vs **Seq2Seq with Attention** for multi-step time series forecasting and see if attention improves accuracy and interpretability.

### Dataset
- Synthetic multivariate time series of **daily energy demand**
- Features:
  1. Temperature (annual sinusoidal + noise)
  2. Humidity (weekly cycle)
  3. Industrial load (volatility + trend)
- Target: `energy_demand` (weighted mixture)
- Train/validation/test split in **temporal order**.

### Preprocessing
- Feature scaling
- Sliding windows for sequence modeling
- Teacher-forcing for decoder inputs

---

## 🧠 Models Implemented

1. **ARIMA** → Baseline linear model  
2. **Seq2Seq LSTM (Vanilla)** → Encoder-decoder, multi-step forecast  
3. **Seq2Seq + Bahdanau Attention** → Focuses on important time steps, improves accuracy and interpretability

---

## 🛠 Hyperparameters

- Hidden units: 64
- Attention units: 32
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Input steps: 60
- Output steps: 14

---

## 📊 Model Evaluations

### ARIMA Baseline

- RMSE: 8.736  
- MAE: 7.243  
- MAPE: 12.37%  

> ARIMA shows high errors because it only uses the target series and cannot capture multivariate interactions or long-term seasonality.

### Seq2Seq LSTM Baseline

- RMSE: 0.065  
- MAE: 0.052  
- MAPE: 11.19%  

**Model summary:**
Model: "functional_24"
Total params: 38,882
Trainable params: 38,882
Non-trainable params: 0


### Seq2Seq + Attention

- RMSE: 0.066  
- MAE: 0.052  
- MAPE: 10.79%  

**Walk-forward validation results:**
- Baseline: RMSE 0.1019, MAE 0.0831, MAPE 18.52%  
- Attention: RMSE 0.0774, MAE 0.0623, MAPE 13.07%

```
Final Metrics Summary (Test Set)
Model               |     RMSE   |   MAE    |  MAPE
Seq2Seq (Baseline)  |    0.1019  |  0.0831  | 18.52%
Seq2Seq + Attention |    0.0774  |  0.0623  | 13.07%
```

> Attention helps the model focus on important encoder time steps, improving both accuracy and interpretability.

---

## 📈 Visualizations

**Forecast vs True**:  
![Forecast Comparison](images/forecast_comparison_07.png)

**Attention Heatmap**:  
![Attention Heatmap](images/attention_heatmap_05.png)

**Walk-forward Samples**:  
![Walk-forward Samples](images/walk_forward_samples_06.png)

**Error Distribution**:  
![Error Distribution](images/error_distribution_08.png)

---

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/abinithi-ka/attention-seq2seq-time-series.git
cd attention-seq2seq-time-series/code
```

2. Install requirements:

pip install -r requirements.txt

3. Run scripts in order :

python 01_data_generation.py
python 02_preprocessing.py
python 03_baseline_arima.py
python 04_baseline_seq2seq.py
python 05_attention_seq2seq.py
python 06_walk_forward_validation.py
python 07_evaluation.py
python 08_visualization.py

Ensure data/raw and data/processed exist (even as empty placeholders).

🔑 Key Takeaways

Attention improves forecast accuracy in multi-step series

Walk-forward validation gives realistic error estimation

Attention weights provide interpretability

📄 References

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning


---

### ✅ **Step 3: Push the updated README to GitHub**

```bash
git add README.md
git commit -m "Update README with model outputs and images"
git push origin main
```

## Environment
- Python 3.9
- TensorFlow 2.x
- NumPy 1.x
- Scikit-learn
- Matplotlib




