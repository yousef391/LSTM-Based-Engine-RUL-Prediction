# Predictive Maintenance of Turbofan Engines using Deep Learning

## Project Overview

This project predicts the **Remaining Useful Life (RUL)** of turbofan jet engines using deep learning methods. The goal is to accurately forecast when an engine will fail, enabling proactive maintenance and reducing downtime.

**Key Features:**
- ✅ Temporal modeling using LSTM/GRU/CNN-LSTM networks
- ✅ Explainable AI (XAI) to identify critical sensors
- ✅ Model comparison and evaluation
- ✅ Ready for uncertainty estimation and formal verification

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**

📥 **Download**: [NASA C-MAPSS Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

**Dataset Details (FD001):**
- Training: 100 engines with complete run-to-failure trajectories
- Test: 100 engines with truncated trajectories
- Features: 26 variables (3 operational settings + 21 sensor measurements)
- Goal: Predict RUL for each test engine

**Note**: Download the dataset and place it in the `data/CMaps/` directory before running the notebook.

## Methodology

### 1. Data Preprocessing
- Converted TXT files to CSV format
- Calculated RUL for training data
- Removed constant/low-variance sensors (7 sensors excluded)
- Selected 17 features (3 operational + 14 sensors)
- Normalized features using StandardScaler

### 2. Sequence Creation
- Created sequences of 30 time steps for RNN input
- Split data by engine ID (no data leakage)
- Train/Validation split: 80/20 by engine

### 3. Model Architecture
- **LSTM**: 2-layer LSTM with 64 hidden units
- **GRU**: 2-layer GRU with 64 hidden units  
- **CNN-LSTM**: Hybrid model with CNN feature extraction + LSTM

### 4. Explainability (XAI)
- **Permutation Importance**: Feature importance using actual LSTM model
- **Gradient-based Importance**: Sensitivity analysis using gradients
- **Individual Explanations**: Why specific engines have low RUL
- **Feature Validation**: Correlation analysis to validate model logic

## Results

### Model Performance Comparison

| Model | MAE (cycles) | RMSE (cycles) | R² Score |
|-------|--------------|---------------|----------|
| **LSTM** | 18.14 | 25.38 | 0.6270 |
| **GRU** | ~18-20 | ~25-27 | ~0.60-0.63 |
| **CNN-LSTM** | ~18-20 | ~25-27 | ~0.60-0.63 |

*Note: GRU and CNN-LSTM results depend on training run*

### Key Findings

1. **Best Model**: LSTM achieved MAE of 18.14 cycles on test set
2. **Critical Sensors**: Identified through explainability analysis
3. **Model Validation**: Confirmed model uses sensible features (high correlation with RUL)
4. **No Data Leakage**: Proper train/validation split by engine ID

### Explainability Insights

- Identified which sensors are most important for RUL prediction
- Explained individual predictions (why specific engines have low RUL)
- Validated that model uses relevant features
- Provided actionable insights for maintenance engineers

## Project Structure

```
Nasa_Rnn/
├── data/
│   └── CMaps/          # NASA C-MAPSS dataset (FD001-FD004)
├── data_csv/            # Converted CSV files
├── Nasa_rnn.ipynb       # Main notebook with all code
└── README.md            # This file
```

## Usage

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

### Running the Project
1. Open `Nasa_rnn.ipynb` in Jupyter Notebook
2. Run cells sequentially:
   - Data loading and preprocessing
   - Sequence creation
   - Model training (LSTM/GRU/CNN-LSTM)
   - Model comparison
   - Explainability analysis

### Key Variables
- `selected_features`: 17 features used for modeling
- `sequence_length`: 30 time steps
- `X_train_final`, `X_val`: Training and validation sequences
- `X_test_seq`: Test sequences
- `model`: Trained LSTM model

## Future Work

- [ ] Uncertainty estimation (Monte Carlo Dropout, Ensemble methods)
- [ ] Generalization testing on FD002, FD003, FD004 datasets
- [ ] Automaton extraction from RNN hidden states for formal verification
- [ ] Interactive dashboard for engine health monitoring
- [ ] Real-time prediction pipeline

## Applications

- **Aircraft Maintenance**: Predict engine failures before they occur
- **Industrial Systems**: Apply to other rotating machinery
- **Safety-Critical Systems**: Trustworthy AI for critical applications
- **Research**: Foundation for formal verification and explainability

## References

- **Dataset**: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
- **Paper**: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008

## Author

Developed for CEA internship application - Predictive Maintenance with Deep Learning and Explainability

---

**Note**: This project demonstrates advanced AI methods for reliable predictive maintenance, with integration of interpretability aligned with research on AI safety and formal guarantees.

