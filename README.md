# Deep Learning — ANN & CNN Projects

A collection of Jupyter notebooks covering fundamentals and comparisons of Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), and classic Perceptron on tabular and image datasets.

All code is Colab-ready (`python3` kernel) and uses `scikit-learn` + `TensorFlow/Keras`.

## 📁 Project Structure

```
DL/
├── ANN_Project.ipynb   # Perceptron vs ANN on Iris (tabular, 4 features, 3 classes)
├── CNN.ipynb           # Perceptron vs ANN vs CNN on MNIST (28x28 images, 10 classes)
├── DL_Model.ipynb      # Minimal binary ANN on synthetic plant-watering data
└── README.md           # this file
```

> **Required external data (not included in repo):**
> * `CNN.ipynb` expects `mnist_train.csv` (60,000 rows) and `mnist_test.csv` (10,000 rows) with columns `label` + `1x1 ... 28x28` in the same directory as the notebook. Download from Kaggle: [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

---

## 1. `ANN_Project.ipynb` — Iris Classification

**Dataset:** `seaborn.load_dataset('iris')` — 150 samples, 4 numeric features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`), 3 balanced classes (`setosa`, `versicolor`, `virginica` — 50 each). `ANN_Project.ipynb:55`

**Pipeline:**

1.  Exploratory Data Analysis: `df.head()` `ANN_Project.ipynb:66`, `value_counts()` `ANN_Project.ipynb:258`, `sns.pairplot` `ANN_Project.ipynb:332`.
2.  Split: `X = df.drop(columns=['species'])`, `y = df['species']` `ANN_Project.ipynb:369`; `LabelEncoder` `ANN_Project.ipynb:381`; `train_test_split(X, y, test_size=0.2, stratify=y_int, random_state=42)` `ANN_Project.ipynb:393`.
3.  Scaling: `StandardScaler` on train/test `ANN_Project.ipynb:404` — **note:** notebook currently calls `fit_transform` on *both* splits; best practice is `transform` on the test set.
4.  **Perceptron baseline:** `sklearn.linear_model.Perceptron(max_iter=1000)` `ANN_Project.ipynb:426` — accuracy **0.90** on 30 test samples `ANN_Project.ipynb:873`, confusion matrix / classification report included `ANN_Project.ipynb:899`.
5.  **ANN (Keras):**

    ```python
    Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    # compile: adam + categorical_crossentropy  ANN_Project.ipynb:952
    # fit: epochs=100, batch_size=8, validation_split=0.2  ANN_Project.ipynb:978
    # y one-hot via to_categorical  ANN_Project.ipynb:940
    ```

    Final validation accuracy ~1.00 (epoch 60+), test accuracy **0.9667** `ANN_Project.ipynb:1200`.

6.  Visualization: training accuracy/loss curves `ANN_Project.ipynb:1224`.

**Key learning:** tabular ANN outperforms linear Perceptron on non-linear Iris boundaries; importance of standardization and stratified splitting.

---

## 2. `CNN.ipynb` — MNIST: Perceptron vs ANN vs CNN

The most comprehensive notebook — systematically compares three model families on the same MNIST data and visualizes the gap.

**Dataset:** `mnist_train.csv` / `mnist_test.csv` — `CNN.ipynb:56` — 784 pixel columns (`1x1`–`28x28`) + `label` (0-9). Train: `(60000, 785)` `CNN.ipynb:500`, Test: `(10000, 785)`. No missing values `CNN.ipynb:575`. Unscaled `int64` 0–255.

**Preprocessing `CNN.ipynb:686`:**

```python
X_train = df.drop("label",axis=1).values / 255.0   # float32 normalization
X_test  similarly
X_train_img = X_train.reshape(-1,28,28)            # for Perceptron/ANN
X_train_cnn = X_train.reshape(-1,28,28,1)          # for CNN
y_*_cat = to_categorical(y_*, 10)
```

**Models:**

| Model | Architecture `CNN.ipynb` | Optimizer | Epochs | Test Acc |
|-------|--------------------------|-----------|--------|----------|
| **Perceptron (single-layer NN)** | `Flatten(28,28) -> Dense(10, softmax)` `CNN.ipynb:736` | `sgd` | 5, bs=32 | **0.8808** `CNN.ipynb:819` |
| **ANN** | `Flatten -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)` `CNN.ipynb:831` | `adam` | 5 | **0.9597** `CNN.ipynb:916` |
| **CNN** | `Conv2D(32,3x3,relu) -> MaxPool(2x2) -> Conv2D(64,3x3,relu) -> MaxPool(2x2) -> Flatten -> Dense(128,relu) -> Dropout(0.5) -> Dense(10,softmax)` `CNN.ipynb:939` | `adam` | 5 | **0.9876** `CNN.ipynb:1028` |

Training logs included for each (`history_percp` `CNN.ipynb:775`, `history_ann` `CNN.ipynb:874`, `history_cnn` `CNN.ipynb:986`).

**Visualizations:**

* `plot_training(history, title)` — per-model accuracy/loss `CNN.ipynb:1039`
* Overlay validation-accuracy comparison `CNN.ipynb:1144`
* `show_side_by_side(models, ...)` — random samples with true vs predicted labels for all three models `CNN.ipynb:1179`

**Key learning:** CNN spatial feature extraction (`Conv2D` + `MaxPooling2D`) dominates on image data; Dropout mitigates overfitting. Clear progression 88% → 96% → 98.7%.

---

## 3. `DL_Model.ipynb` — Toy Binary Classifier (Plant Watering)

A minimal end-to-end DL example on handcrafted data to illustrate normalization and binary classification.

**Dataset:** Synthetic 16-row DataFrame `DL_Model.ipynb:37`:

| feature | meaning |
|---------|---------|
| `soil_moisture` | 0.05–0.80 |
| `temperature_c` | 19–35 |
| `sunlight_hours` | 1–12 |
| `needs_water` | target 0/1 (8 pos / 8 neg) |

**Pipeline:**

1.  Inspect `df.columns` `DL_Model.ipynb:372`, DataFrame display `DL_Model.ipynb:44`.
2.  **Normalization (min-max):** `X_scaled = (X - X.min())/(X.max()-X.min()+1e-8)` `DL_Model.ipynb:413` — noted as DL convention vs ML standardization `DL_Model.ipynb:398`.
3.  Split: `train_test_split(X, y, test_size=0.25, random_state=42)` `DL_Model.ipynb:727` — **note:** split uses raw `X`, not `X_scaled`; you may want to flow `X_scaled` instead.
4.  **Model `DL_Model.ipynb:747`:**

    ```python
    keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train.values, y_train.values, validation_data=(X_test.values, y_test.values),
              epochs=100, batch_size=1)
    ```

5.  Training for 100 epochs — accuracy ~0.58–0.75 oscillating due to tiny dataset and SGD with batch_size=1 `DL_Model.ipynb:783`. Demonstrates why "more data is better" for deep learning.
6.  Empty import cell for `optimizers` `DL_Model.ipynb:1009` left for experimentation.

**Key learning:** `sigmoid` + `binary_crossentropy` for binary targets, `Input` layer, min-max scaling, effects of extremely small data on generalization.

---

## 🧰 Tech Stack

* Python 3 (Colab)
* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `scikit-learn` — `Perceptron`, `LabelEncoder`, `StandardScaler`, `train_test_split`, `accuracy_score`, `classification_report`
* `tensorflow` / `keras` — `Sequential`, `Dense`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, `to_categorical`

---

## ⚙️ Installation & Setup

```bash
# clone / copy the DL folder locally
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

# optional: jupyter
pip install notebook
jupyter notebook
```

Then open any `*.ipynb` in Jupyter, VS Code, or upload to Google Colab.

> For `CNN.ipynb`, place `mnist_train.csv` and `mnist_test.csv` alongside the notebook before running.

---

## ▶️ How to Run

1.  Open `ANN_Project.ipynb` → Run all cells top-to-bottom. No external files needed.
2.  Open `CNN.ipynb` → Ensure MNIST CSVs are present → Run all. GPU runtime recommended (CNN: ~80 s/epoch on CPU, faster on Colab T4).
3.  Open `DL_Model.ipynb` → Run all. No dependencies beyond `pandas`/`tensorflow`.

---

## 📊 Results Summary

| Notebook | Task | Best Model | Accuracy |
|----------|------|------------|----------|
| ANN_Project | Iris (3-class) | ANN 16-8 softmax | 96.7% test |
| ANN_Project | Iris baseline | Perceptron | 90.0% test |
| CNN | MNIST (10-class) | CNN Conv32/64 | **98.76%** |
| CNN | MNIST | ANN 128-64 | 95.97% |
| CNN | MNIST baseline | Perceptron (logistic) | 88.08% |
| DL_Model | Plant watering (binary) | Dense 8 → sigmoid | ~66% (tiny data) |

---

## 💡 Concepts Demonstrated

* Perceptron vs multi-layer ANN vs CNN capacity
* Tabular (`StandardScaler`) vs image (`/255` + `reshape`) preprocessing
* One-hot encoding (`to_categorical`), `softmax`/`categorical_crossentropy` vs `sigmoid`/`binary_crossentropy`
* `Conv2D` feature extraction, `MaxPooling2D` down-sampling, `Dropout` regularization
* Train/validation splits, `validation_split` vs `validation_data`, learning curves

---

## 🔧 Known Quirks & Suggested Fixes

* `ANN_Project.ipynb:404` — change `scaler.fit_transform(X_test)` → `scaler.transform(X_test)` to avoid leakage.
* `CNN.ipynb:775` high loss values (260) for Perceptron — loss not normalized; still converges but check `from_logits` usage.
* `DL_Model.ipynb:727` — split currently uses unscaled `X`; switch to `X_scaled` for proper normalization benefit.
* `CNN.ipynb:56` hard-coded CSV paths — consider `tensorflow.keras.datasets.mnist.load_data()` for portability.

---

## 📄 License & Author

For educational use. Created as a Deep Learning lab project covering ANN and CNN fundamentals.

* Environment: Google Colab, `nbformat 4`
* To reproduce, run cells sequentially; random states are fixed where specified (`42`).

---

*Generated for `C:\Users\DELL\Downloads\DL` — all references are local to this folder as requested.*
