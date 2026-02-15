# 🔥 PyroSense — Forest Fire FWI Prediction System

A machine learning web application that predicts the **Fire Weather Index (FWI)** for forest fire risk assessment, trained on the Algerian Forest Fire dataset.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Demo](#-demo)
3. [How It Works](#-how-it-works)
4. [Dataset](#-dataset)
5. [Machine Learning Model](#-machine-learning-model)
6. [Input Parameters](#-input-parameters)
7. [FWI Risk Levels](#-fwi-risk-levels)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [Running the App](#-running-the-app)
11. [API Routes](#-api-routes)
12. [Technologies Used](#-technologies-used)
13. [Screenshots](#-screenshots)
14. [Troubleshooting](#-troubleshooting)

---

## 🌟 Project Overview

**PyroSense** is a Flask-based web application that uses a trained Ridge Regression model to predict the Fire Weather Index (FWI) — a numeric score that quantifies fire danger based on weather and fuel moisture conditions.

The app takes **15 input parameters** (date, weather conditions, FWI system indices, fire classification, and region) and returns a predicted FWI score along with a risk level classification.

### Key Features
- 🎯 Real-time FWI prediction using Ridge Regression
- 🌍 Covers 2 Algerian regions: Bejaia and Sidi-Bel Abbes
- 📊 Risk level classification (Low → Extreme)
- 🎨 Modern, responsive UI with fire-themed design
- ⚡ Sub-second prediction time

---

## 🎬 Demo

| Page | URL | Description |
|------|-----|-------------|
| Landing Page | `http://localhost/` | Introduction and project overview |
| Predictor | `http://localhost/predict` | Fill form and get FWI prediction |

---

## ⚙️ How It Works

```
User fills form (15 parameters)
        ↓
Flask receives POST request
        ↓
Values extracted from form data
        ↓
Pandas DataFrame created with correct column order
        ↓
StandardScaler normalizes the input
        ↓
Ridge Regression model predicts FWI score
        ↓
Result displayed with risk classification
```

### Step-by-Step Flow

1. **User visits** `http://localhost/` → sees the landing page (`index.html`)
2. **User clicks** "Launch FWI Predictor" → routed to `/predict` (`home.html`)
3. **User fills** all 15 input fields and clicks "Predict FWI"
4. **Flask processes** the POST request in `app.py`:
   - Reads form values
   - Encodes `Classes` as `1` (fire) or `0` (not fire)
   - Builds a `pandas DataFrame` with the exact column order the scaler was trained on
   - Applies `StandardScaler.transform()` to normalize values
   - Passes scaled array to `ridge_model.predict()`
5. **Result rendered** back on `home.html` with score + risk level + recommendations

---

## 📦 Dataset

**Name:** Algerian Forest Fires Dataset  
**Source:** UCI Machine Learning Repository  
**Records:** 244 instances (122 per region)  
**Period:** June 2012 – September 2012  

### Regions
| Code | Region | Location |
|------|--------|----------|
| `0` | Bejaia | Northeast Algeria |
| `1` | Sidi-Bel Abbes | Northwest Algeria |

---

## 🤖 Machine Learning Model

### Algorithm: Ridge Regression
Ridge Regression is a linear model with L2 regularization that prevents overfitting by penalizing large coefficients.

```
FWI_predicted = β₀ + β₁·day + β₂·month + ... + β₁₅·Region + λ·‖β‖²
```

Where `λ` is the regularization strength (alpha).

### Preprocessing
- **StandardScaler** — normalizes all features to zero mean and unit variance
- This is critical because features like `DC` (0–800+) and `Region` (0 or 1) have very different scales

### Model Files
| File | Description |
|------|-------------|
| `models/ridge.pkl` | Trained Ridge Regression model |
| `models/scaler.pkl` | Fitted StandardScaler |

### Why Ridge Regression?
- Works well on correlated features (FWI components are correlated by definition)
- Resistant to overfitting on small datasets (244 samples)
- Fast inference — predictions in microseconds

---

## 📊 Input Parameters

The model requires exactly **15 parameters**:

### Group 1 — Date (3 fields)
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `day` | Integer | 1–31 | Day of the observation |
| `month` | Integer | 1–12 | Month of the observation |
| `year` | Integer | 2000–2100 | Year of the observation |

### Group 2 — Weather Conditions (4 fields)
| Parameter | Unit | Description |
|-----------|------|-------------|
| `Temperature` | °C | Noon air temperature |
| `RH` | % | Relative humidity at noon |
| `Ws` | km/h | Wind speed at noon |
| `Rain` | mm | Total daily rainfall |

### Group 3 — FWI System Components (6 fields)
| Parameter | Full Name | Description |
|-----------|-----------|-------------|
| `FFMC` | Fine Fuel Moisture Code | Moisture content of litter and fine fuels (0–101) |
| `DMC` | Duff Moisture Code | Average moisture of loosely compacted organic layers |
| `DC` | Drought Code | Moisture content of deep compact organic layers |
| `ISI` | Initial Spread Index | Expected rate of fire spread |
| `BUI` | Build Up Index | Amount of fuel available for combustion |
| `FWI` | Fire Weather Index | Overall fire intensity rating |

### Group 4 — Classification & Region (2 fields)
| Parameter | Values | Description |
|-----------|--------|-------------|
| `Classes` | `fire` / `not fire` | Observed fire classification (encoded: fire=1, not fire=0) |
| `Region` | `0` / `1` | Algerian region (0=Bejaia, 1=Sidi-Bel Abbes) |

---

## 🚦 FWI Risk Levels

| FWI Range | Risk Level | Meaning |
|-----------|------------|---------|
| 0 – 5.2 | 🟢 **Low** | Fires unlikely to start or spread |
| 5.2 – 11.2 | 🔵 **Moderate** | Fires can start under dry conditions |
| 11.2 – 21.3 | 🟡 **High** | Fires start easily and spread rapidly |
| 21.3 – 38 | 🟠 **Very High** | All fires are potentially serious |
| 38+ | 🔴 **Extreme** | Any fire will spread rapidly and be uncontrollable |

---

## 📁 Project Structure

```
pyrosense/
│
├── app.py                  # Main Flask application
│
├── models/
│   ├── ridge.pkl           # Trained Ridge Regression model
│   └── scaler.pkl          # Fitted StandardScaler
│
├── templates/
│   ├── index.html          # Landing page
│   └── home.html           # Prediction form + result page
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone or download the project
```bash
git clone https://github.com/yourusername/pyrosense.git
cd pyrosense
```

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
flask
numpy
pandas
scikit-learn
```

### 4. Ensure model files exist
```
models/
├── ridge.pkl    ← must exist
└── scaler.pkl   ← must exist
```

---

## 🚀 Running the App

### Option A — Port 80 (recommended on institutional networks)
```bash
# Run as Administrator (Windows) or sudo (Linux/Mac) — port 80 needs elevated privileges
python app.py
```
Access at: **http://localhost** or **http://your-ip-address**

### Option B — Port 5000 (local development)
Change `port=80` to `port=5000` in `app.py`:
```python
app.run(host="0.0.0.0", port=5000, debug=True)
```
Access at: **http://localhost:5000**

### Option C — Running on Windows without admin rights
Use port 8080 or 8000:
```python
app.run(host="0.0.0.0", port=8080, debug=True)
```
Access at: **http://localhost:8080**

---

## 🌐 API Routes

| Method | Route | Template | Description |
|--------|-------|----------|-------------|
| `GET` | `/` | `index.html` | Landing page |
| `GET` | `/predict` | `home.html` | Show empty prediction form |
| `POST` | `/predict` | `home.html` | Process form, return prediction |

### POST /predict — Expected Form Fields

```
day, month, year, Temperature, RH, Ws, Rain,
FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region
```

### Example Test Values (from dataset — low risk day)

| Field | Value |
|-------|-------|
| day | 1 |
| month | 6 |
| year | 2012 |
| Temperature | 29 |
| RH | 57 |
| Ws | 18 |
| Rain | 0.0 |
| FFMC | 65.7 |
| DMC | 3.4 |
| DC | 7.6 |
| ISI | 1.3 |
| BUI | 3.4 |
| FWI | 0.5 |
| Classes | not fire |
| Region | 0 |

---

## 🧰 Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Backend language |
| Flask | 2.x | Web framework |
| scikit-learn | 1.x | ML model & scaler |
| NumPy | 1.x | Numerical operations |
| Pandas | 1.x | DataFrame for feature names |
| HTML5/CSS3 | — | Frontend templates |
| JavaScript | ES6+ | Form interactivity |
| Google Fonts | — | Typography (Bebas Neue, Plus Jakarta Sans, DM Mono) |

---

## 🔧 Troubleshooting

### ❌ 404 Not Found
- Make sure you're visiting `/` or `/predict`, not a `.html` file directly
- Confirm `app.py` is running and you see `* Running on http://...` in terminal
- Check the port — if running on port 80, visit `http://localhost` (no port number)

### ❌ 500 Internal Server Error
- Check that `models/ridge.pkl` and `models/scaler.pkl` exist
- Ensure all dependencies are installed: `pip install flask numpy pandas scikit-learn`

### ❌ Feature names mismatch warning / error
- This happens when the scaler was trained with named pandas columns but receives a plain numpy array
- Fixed in `app.py` by passing `pd.DataFrame([data], columns=FEATURE_COLUMNS)`
- `FEATURE_COLUMNS` is auto-read from `scaler.feature_names_in_` at startup

### ❌ Gateway Timeout / Page Cannot Be Displayed
- Your network/firewall is blocking the port
- Switch to port 80: `app.run(port=80)` and run as Administrator
- Or try port 8080: `app.run(port=8080)`

### ❌ sklearn UserWarning about feature names
- Safe to ignore — prediction still works correctly
- Fully resolved by using `pd.DataFrame` instead of `np.array` (already fixed)

---

## 📝 License

This project is for educational purposes. The dataset is from the UCI Machine Learning Repository.
BY -> Nilesh Nayak
---

*Built with Flask · Scikit-learn · Algerian Forest Fire Dataset*
