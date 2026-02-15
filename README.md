# 🔥 PyroSense — Forest Fire FWI Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Render](https://img.shields.io/badge/Hosted_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**A machine learning web app that predicts Forest Fire Weather Index (FWI) in real-time using Ridge Regression — trained on the Algerian Forest Fire Dataset.**

[🚀 Live Demo](https://pyrosense.onrender.com) · [📊 Dataset](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) · [🐛 Report Bug](https://github.com/nilesh5566/pyrosense/issues)

</div>

---

## 📋 Table of Contents

1. [What is This Project?](#-what-is-this-project)
2. [What is FWI?](#-what-is-fwi-fire-weather-index)
3. [Live Demo](#-live-demo)
4. [Architecture](#-architecture)
5. [How It Works — Full Flow](#-how-it-works--full-flow)
6. [Dataset](#-dataset)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Input Parameters](#-input-parameters-explained)
9. [FWI Risk Levels](#-fwi-risk-levels)
10. [Project Structure](#-project-structure)
11. [Pages & UI](#-pages--ui)
12. [API Reference](#-api-reference)
13. [Installation & Setup](#-installation--setup)
14. [Deployment on Render](#-deployment-on-render)
15. [Technologies Used](#-technologies-used)
16. [Troubleshooting](#-troubleshooting)
17. [Future Improvements](#-future-improvements)

---

## 🌟 What is This Project?

**PyroSense** is a full-stack machine learning web application built with **Python + Flask** that predicts how dangerous a forest fire situation is — before the fire even starts.

It works by taking **15 real-world weather and fire index measurements**, running them through a trained **Ridge Regression model**, and returning a **Fire Weather Index (FWI) score** — a globally recognized number that tells firefighters and forest rangers how severe the fire risk is on that day.

### Why was this built?
Forest fires in Algeria (and globally) cause enormous damage to ecosystems, property, and lives. By predicting fire danger *ahead of time* using weather data, authorities can:
- Pre-position firefighting resources
- Issue public warnings early
- Restrict human activity in high-risk zones
- Potentially save lives and forests

### Who is it for?
- 🌲 **Forest rangers** — assess daily fire risk
- 🚒 **Fire departments** — resource planning
- 🎓 **Students / Researchers** — ML + Flask project reference
- 💻 **Developers** — example of end-to-end ML deployment

---

## 🔥 What is FWI? (Fire Weather Index)

The **Fire Weather Index (FWI)** is a Canadian-developed system adopted worldwide to measure fire danger. It is calculated from weather observations and produces a single number representing potential fire intensity.

### The FWI System Components

```
                    WEATHER INPUTS
              ┌─────────────────────────┐
              │  Temperature  Rain      │
              │  Humidity     Wind      │
              └────────┬────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  FFMC   │   │   DMC   │   │   DC    │
   │ Fine    │   │  Duff   │   │Drought  │
   │  Fuel   │   │Moisture │   │  Code   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        ▼              └──────┬───────┘
   ┌─────────┐                ▼
   │   ISI   │          ┌─────────┐
   │ Initial │          │   BUI   │
   │ Spread  │          │Build Up │
   │  Index  │          │  Index  │
   └────┬────┘          └────┬────┘
        │                    │
        └──────────┬──────────┘
                   ▼
            ┌────────────┐
            │    FWI     │
            │Fire Weather│
            │   Index    │
            └────────────┘
```

| Code | Full Name | What It Measures |
|------|-----------|-----------------|
| **FFMC** | Fine Fuel Moisture Code | Moisture in dry leaves, grass, and litter on forest floor |
| **DMC** | Duff Moisture Code | Moisture in loosely compacted organic matter |
| **DC** | Drought Code | Moisture deep in the soil and thick logs |
| **ISI** | Initial Spread Index | How fast a fire would spread (wind + FFMC) |
| **BUI** | Build Up Index | Total fuel available (DMC + DC) |
| **FWI** | Fire Weather Index | Final danger rating combining ISI + BUI |

**Higher FWI = More dangerous fire conditions.**

---

## 🎬 Live Demo

| Page | URL | Description |
|------|-----|-------------|
| 🏠 Landing Page | `https://pyrosense.onrender.com/` | Project overview with live stats |
| 🔮 Predictor | `https://pyrosense.onrender.com/predict` | Fill 15 parameters → get FWI |

### Test it instantly with this sample input:

| Field | Value | Field | Value |
|-------|-------|-------|-------|
| Day | `1` | FFMC | `65.7` |
| Month | `6` | DMC | `3.4` |
| Year | `2012` | DC | `7.6` |
| Temperature | `29` | ISI | `1.3` |
| Humidity (RH) | `57` | BUI | `3.4` |
| Wind Speed | `18` | FWI | `0.5` |
| Rain | `0.0` | Classes | `not fire` |
| Region | `0 (Bejaia)` | | |

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                                                             │
│  ┌─────────────┐    click    ┌─────────────────────────┐   │
│  │  index.html │ ──────────► │       home.html          │   │
│  │ Landing Page│             │  Prediction Form + Result│   │
│  └─────────────┘             └──────────┬──────────────┘   │
└────────────────────────────────────────-│───────────────────┘
                                          │ HTTP POST /predict
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FLASK SERVER (app.py)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Route: /predict                     │  │
│  │                                                       │  │
│  │  1. Extract 15 form values                            │  │
│  │  2. Encode: Classes → 1/0                             │  │
│  │  3. Build pd.DataFrame with column names              │  │
│  │  4. scaler.transform(df)  ──► StandardScaler          │  │
│  │  5. ridge_model.predict(scaled) ──► FWI Score         │  │
│  │  6. render_template('home.html', prediction_text=...) │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │  models/     │    │  templates/                       │  │
│  │  ridge.pkl   │    │  index.html  (landing page)       │  │
│  │  scaler.pkl  │    │  home.html   (form + result)      │  │
│  └──────────────┘    └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Hosted on
                              ▼
                    ┌──────────────────┐
                    │   Render.com     │
                    │  (Free Hosting)  │
                    │  gunicorn server │
                    └──────────────────┘
```

### ML Pipeline Architecture

```
  RAW FORM INPUT (15 values)
           │
           ▼
  ┌─────────────────────┐
  │   Data Extraction   │
  │  request.form[key]  │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    Encoding         │
  │  Classes:           │
  │  "fire"    → 1      │
  │  "not fire"→ 0      │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Pandas DataFrame   │
  │  (preserves column  │
  │   names for scaler) │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  StandardScaler     │
  │  .transform()       │
  │                     │
  │  x_scaled =         │
  │  (x - mean) / std   │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Ridge Regression   │
  │  .predict()         │
  │                     │
  │  FWI = β₀ + Σβᵢxᵢ  │
  │      + λ‖β‖²        │
  └──────────┬──────────┘
             │
             ▼
     FWI SCORE (float)
```

---

## ⚙️ How It Works — Full Flow

Here is exactly what happens from the moment a user opens the app to getting a prediction:

```
STEP 1: User opens https://pyrosense.onrender.com
        │
        └─► Flask serves index.html (landing page)
            Shows: hero section, stats, features, canvas fire animation

STEP 2: User clicks "Launch FWI Predictor"
        │
        └─► Browser sends GET /predict
            Flask serves home.html with empty form

STEP 3: User fills all 15 input fields
        │
        ├─ Section 1: Day, Month, Year, Temp, RH, Wind, Rain
        ├─ Section 2: FFMC, DMC, DC, ISI, BUI, FWI
        └─ Section 3: Classes (dropdown), Region (dropdown)

STEP 4: User clicks "Predict FWI"
        │
        └─► Browser sends POST /predict with form data

STEP 5: Flask app.py processes the request
        │
        ├─ Reads all 15 values from request.form
        ├─ Converts strings to int/float
        ├─ Encodes "fire" → 1, "not fire" → 0
        ├─ Creates pd.DataFrame with EXACT column order
        ├─ Calls scaler.transform(df) → normalized array
        ├─ Calls ridge_model.predict(scaled) → FWI float
        └─ Returns render_template('home.html', prediction_text=...)

STEP 6: home.html displays the result
        │
        ├─ Shows FWI score in large text
        ├─ Classifies risk level (Low/Moderate/High/Very High/Extreme)
        ├─ Fills animated gauge bar proportionally
        └─ Shows description + recommendations
```

---

## 📦 Dataset

### Algerian Forest Fires Dataset

| Property | Value |
|----------|-------|
| **Name** | Algerian Forest Fires Dataset |
| **Source** | UCI Machine Learning Repository |
| **Year** | 2019 |
| **Records** | 244 instances |
| **Features** | 13 attributes + class |
| **Period** | June 2012 – September 2012 |
| **Task type** | Regression (FWI prediction) |

### Two Regions Covered

```
ALGERIA
┌──────────────────────────────────────────────────┐
│                                                  │
│   ┌──────────────────┐   ┌─────────────────────┐ │
│   │    BEJAIA        │   │  SIDI-BEL ABBES     │ │
│   │  Region 0        │   │  Region 1           │ │
│   │  Northeast       │   │  Northwest          │ │
│   │  122 records     │   │  122 records        │ │
│   │                  │   │                     │ │
│   │  Near coast      │   │  Interior/semi-arid │ │
│   │  Mediterranean   │   │  Hotter summers     │ │
│   └──────────────────┘   └─────────────────────┘ │
└──────────────────────────────────────────────────┘
```

### Dataset Sample

```
day  month  year  Temp  RH  Ws  Rain  FFMC   DMC    DC   ISI  BUI   FWI  Classes   Region
 1     6   2012    29   57  18   0.0  65.7   3.4   7.6   1.3  3.4   0.5  not fire    0
  2     6   2012    29   61  13   1.3  64.7   4.1   7.6   1.0  3.9   0.4  not fire    0
  3     6   2012    26   82   22   13.1  47.1   2.5   7.1   0.3  2.7   0.1  not fire    0
```

---

## 🤖 Machine Learning Pipeline

### Algorithm: Ridge Regression

**Ridge Regression** is a linear regression model with **L2 regularization**. It adds a penalty to the loss function to prevent overfitting:

```
Standard Linear Regression:
  Loss = Σ(yᵢ - ŷᵢ)²

Ridge Regression adds L2 penalty:
  Loss = Σ(yᵢ - ŷᵢ)² + λ · Σβⱼ²
                         └─────────┘
                          L2 penalty
                     (shrinks coefficients)

Final equation:
  FWI = β₀ + β₁·day + β₂·month + β₃·year
          + β₄·Temp + β₅·RH + β₆·Ws + β₇·Rain
          + β₈·FFMC + β₉·DMC + β₁₀·DC
          + β₁₁·ISI + β₁₂·BUI + β₁₃·FWI
          + β₁₄·Classes + β₁₅·Region
          + λ·‖β‖²
```

Where:
- `β₀` = intercept (bias)
- `β₁...β₁₅` = learned coefficients for each feature
- `λ` = regularization strength (alpha hyperparameter)
- `‖β‖²` = sum of squared coefficients (L2 norm)

### Why Ridge Regression?

| Reason | Explanation |
|--------|-------------|
| **Correlated features** | FWI components (FFMC, DMC, DC etc.) are mathematically related — Ridge handles this better than plain linear regression |
| **Small dataset** | Only 244 samples — Ridge prevents overfitting via regularization |
| **Interpretable** | Coefficients show which features matter most |
| **Fast** | Predictions take microseconds — ideal for web apps |
| **Stable** | Less sensitive to outliers than other models |

### Preprocessing: StandardScaler

Before feeding data to the model, every feature is **normalized**:

```
For each feature x:
  x_scaled = (x - mean) / standard_deviation

Example:
  Temperature values: [20, 29, 35, 42]
  Mean = 31.5, Std = 8.1

  20 → (20 - 31.5) / 8.1 = -1.42
  29 → (29 - 31.5) / 8.1 = -0.31
  35 → (35 - 31.5) / 8.1 =  0.43
  42 → (42 - 31.5) / 8.1 =  1.30
```

This is essential because features like:
- `DC` can reach 800+
- `Region` is only 0 or 1
- `Rain` is usually 0.0

Without scaling, large-value features would dominate the model unfairly.

### Model Training Summary

```
Dataset (244 rows)
      │
      ├─ 80% Training set (195 rows)
      └─ 20% Test set (49 rows)
            │
            ▼
   StandardScaler.fit(X_train)    ← learns mean & std from training data
            │
            ▼
   StandardScaler.transform(X)    ← applies to both train and test
            │
            ▼
   RidgeRegression.fit(X_scaled, y)   ← learns coefficients
            │
            ▼
   ridge.pkl + scaler.pkl saved with pickle
```

---

## 📊 Input Parameters Explained

The model requires exactly **15 parameters** split into 4 groups:

### Group 1 — Date (3 parameters)

| Parameter | Type | Range | Example | Why needed |
|-----------|------|-------|---------|------------|
| `day` | Integer | 1–31 | `15` | Seasonal fire patterns vary by day |
| `month` | Integer | 1–12 | `6` | Summer months = higher risk |
| `year` | Integer | 2000+ | `2012` | Long-term climate trends |

### Group 2 — Weather Conditions (4 parameters)

| Parameter | Unit | Range | Example | Effect on Fire |
|-----------|------|-------|---------|---------------|
| `Temperature` | °C | 22–42 | `29` | ↑ Temp = ↑ fuel dryness |
| `RH` | % | 21–90 | `57` | ↓ Humidity = ↑ fire risk |
| `Ws` (Wind Speed) | km/h | 6–29 | `18` | ↑ Wind = ↑ spread rate |
| `Rain` | mm | 0–16.8 | `0.0` | ↑ Rain = ↓ fire risk |

### Group 3 — FWI System Codes (6 parameters)

| Parameter | Full Name | Range | Example | Description |
|-----------|-----------|-------|---------|-------------|
| `FFMC` | Fine Fuel Moisture Code | 28.6–96.2 | `65.7` | Moisture of leaves & grass on forest floor. High = very dry = easy ignition |
| `DMC` | Duff Moisture Code | 1.1–65.9 | `3.4` | Moisture of loosely packed organic layers (10–15cm deep) |
| `DC` | Drought Code | 7–220.4 | `7.6` | Moisture deep in soil and thick logs. Reflects long drought periods |
| `ISI` | Initial Spread Index | 0–18.5 | `1.3` | How fast fire would spread. Combines wind speed and FFMC |
| `BUI` | Build Up Index | 1.1–68 | `3.4` | Total fuel available. Combines DMC and DC |
| `FWI` | Fire Weather Index | 0–31.1 | `0.5` | Overall fire danger. Combines ISI and BUI |

### Group 4 — Classification & Region (2 parameters)

| Parameter | Options | Encoded As | Description |
|-----------|---------|------------|-------------|
| `Classes` | `fire` / `not fire` | `1` / `0` | Was there actually a fire that day? |
| `Region` | `Bejaia` / `Sidi-Bel Abbes` | `0` / `1` | Which Algerian region |

---

## 🚦 FWI Risk Levels

The predicted FWI score maps to these internationally recognized danger levels:

```
FWI Scale:
0─────────5.2─────────11.2─────────21.3─────────38──────────►
│   LOW   │  MODERATE │   HIGH    │  VERY HIGH  │  EXTREME  │
│  🟢     │    🔵     │    🟡     │     🟠      │    🔴     │
└─────────┴───────────┴───────────┴─────────────┴───────────┘
```

| Range | Level | What it means | Recommended Action |
|-------|-------|--------------|-------------------|
| 0 – 5.2 | 🟢 **Low** | Fires unlikely to start or spread | Normal monitoring |
| 5.2 – 11.2 | 🔵 **Moderate** | Fires can start under dry conditions | Increased vigilance |
| 11.2 – 21.3 | 🟡 **High** | Fires start easily, spread rapidly | Active patrols, restrict burning |
| 21.3 – 38 | 🟠 **Very High** | All fires are potentially serious | Pre-position resources, issue warnings |
| 38+ | 🔴 **Extreme** | Any fire is nearly uncontrollable | Emergency protocols, possible evacuations |

---

## 📁 Project Structure

```
pyrosense/
│
├── 📄 app.py                    # Flask application — all routes & ML logic
│
├── 📄 requirements.txt          # Python package dependencies
│
├── 📄 render.yaml               # Render.com deployment configuration
│
├── 📄 README.md                 # This documentation file
│
├── 📂 models/                   # Trained ML model files
│   ├── 🤖 ridge.pkl             # Ridge Regression model (sklearn)
│   └── ⚖️  scaler.pkl           # StandardScaler (fitted on training data)
│
└── 📂 templates/                # HTML pages (Jinja2 templates)
    ├── 🏠 index.html            # Landing page with fire animation
    └── 🔮 home.html             # Prediction form + result display
```

### What each file does

**`app.py`** — The brain of the application
```python
# Loads the ML models at startup
# Defines 2 routes:
#   GET  /          → serves index.html (landing page)
#   GET  /predict   → serves home.html (empty form)
#   POST /predict   → processes form → runs ML → returns result
```

**`models/ridge.pkl`** — The trained Ridge Regression model
```
Saved using pickle after training on Algerian Forest Fire dataset.
Contains learned coefficients (β values) for all 15 features.
```

**`models/scaler.pkl`** — The fitted StandardScaler
```
Saved using pickle after fitting on training data.
Contains mean and std for each of the 15 features.
Also contains feature_names_in_ — the exact column order it expects.
```

**`templates/index.html`** — Landing page
```
Animated fire canvas background
Hero section with headline + CTA button → links to /predict
Stats bar (15 parameters, 98.2% accuracy, etc.)
Feature cards explaining the system
```

**`templates/home.html`** — Prediction page
```
3-section form:
  Section 1: Date + Weather (7 fields)
  Section 2: FWI Components (6 fields)
  Section 3: Classes + Region (2 fields)
Submit button → POST to /predict
Result display: FWI score, risk badge, gauge, recommendations
```

---

## 🖥️ Pages & UI

### Page 1 — Landing Page (`/`)

```
┌─────────────────────────────────────────────────┐
│  🔴 PYROSENSE                      Model Active │ ← Nav
├─────────────────────────────────────────────────┤
│                                                 │
│  PREDICT                  ┌─────────────────┐  │
│  WILDFIRE  ◄──────────    │  FWI: 28.7      │  │  ← Hero
│                           │  ████████░░ HIGH│  │
│  [Launch Predictor →]     │  Temp: 29°      │  │
│                           └─────────────────┘  │
├─────────────────────────────────────────────────┤
│  15 params │ 98.2% acc │ <0.3s │ 244 samples   │ ← Stats
├─────────────────────────────────────────────────┤
│  🌡️ Weather │ 📊 FWI Suite │ 🤖 ML │ ⚡ Speed  │ ← Features
├─────────────────────────────────────────────────┤
│  © 2025 PyroSense                               │ ← Footer
└─────────────────────────────────────────────────┘
```

### Page 2 — Predictor (`/predict`)

```
┌─────────────────────────────────────────────────┐
│  🔴 PYROSENSE    Home / FWI Predictor  ● Active │ ← Nav
├─────────────────────────────────────────────────┤
│  FWI Predictor              ✓Launch › ●Fill › Result│
│  Enter weather data...                          │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ 1 📅 Date & Weather           7 fields  │   │
│  │ ┌──────┐ ┌───────┐ ┌──────┐            │   │
│  │ │ Day  │ │ Month │ │ Year │            │   │
│  │ └──────┘ └───────┘ └──────┘            │   │
│  │ ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐  │   │
│  │ │ Temp │ │  RH   │ │ Wind │ │ Rain │  │   │
│  │ └──────┘ └───────┘ └──────┘ └──────┘  │   │
│  ├─────────────────────────────────────────┤   │
│  │ 2 🔥 FWI Components           6 fields  │   │
│  │ ┌─────┐ ┌─────┐ ┌────┐                │   │
│  │ │FFMC │ │ DMC │ │ DC │                │   │
│  │ └─────┘ └─────┘ └────┘                │   │
│  │ ┌─────┐ ┌─────┐ ┌─────┐              │   │
│  │ │ ISI │ │ BUI │ │ FWI │              │   │
│  │ └─────┘ └─────┘ └─────┘              │   │
│  ├─────────────────────────────────────────┤   │
│  │ 3 📍 Classification & Region  2 fields  │   │
│  │ ┌──────────────┐ ┌──────────────────┐  │   │
│  │ │ 🔥 Fire ▾   │ │ 📍 Bejaia ▾     │  │   │
│  │ └──────────────┘ └──────────────────┘  │   │
│  ├─────────────────────────────────────────┤   │
│  │ 15 params required      [⚡ Predict FWI]│   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │         Predicted FWI Score             │   │ ← Result
│  │              28.7                       │   │
│  │         Fire Weather Index              │   │
│  │         🟠 Very High Risk               │   │
│  │  ████████████████████░░░░ gauge         │   │
│  ├────────────────┬────────────────────────┤   │
│  │ What it means  │ Recommendation         │   │
│  └────────────────┴────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 🌐 API Reference

### Routes

| Method | Endpoint | Template | Description |
|--------|----------|----------|-------------|
| `GET` | `/` | `index.html` | Landing page |
| `GET` | `/predict` | `home.html` | Empty prediction form |
| `POST` | `/predict` | `home.html` | Process inputs → return FWI |

### POST `/predict` — Form Fields

```
Content-Type: application/x-www-form-urlencoded

Required fields:
  day          → integer (1–31)
  month        → integer (1–12)
  year         → integer
  Temperature  → float (°C)
  RH           → float (0–100, %)
  Ws           → float (km/h)
  Rain         → float (mm)
  FFMC         → float (0–101)
  DMC          → float
  DC           → float
  ISI          → float
  BUI          → float
  FWI          → float
  Classes      → string ("fire" or "not fire")
  Region       → integer (0 or 1)
```

### Response

Returns `home.html` with `prediction_text` variable:
```
"Predicted FWI: 28.7"   ← success
"Error: <message>"      ← failure
```

### Internal Processing (app.py)

```python
# 1. Extract from form
data = {
    'day': int(request.form['day']),
    ...
    'Classes': 1 if request.form['Classes'] == 'fire' else 0,
    'Region': int(request.form['Region'])
}

# 2. Create DataFrame (preserves column names for scaler)
FEATURE_COLUMNS = list(scaler.feature_names_in_)   # auto-read at startup
df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

# 3. Scale + Predict
scaled = scaler.transform(df)
fwi    = ridge_model.predict(scaled)[0]
```

---

## 🛠️ Installation & Setup

### Prerequisites

| Tool | Minimum Version | Check with |
|------|----------------|------------|
| Python | 3.8+ | `python --version` |
| pip | 21+ | `pip --version` |
| Git | 2.x | `git --version` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/pyrosense.git
cd pyrosense
```

### Step 2 — Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

> A virtual environment keeps project dependencies isolated from your system Python.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
flask
numpy
pandas
scikit-learn
gunicorn
```

### Step 4 — Verify models exist

```bash
ls models/
# Should show: ridge.pkl  scaler.pkl
```

### Step 5 — Run locally

```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🚀 Deployment on Render

### Required files

**`render.yaml`**
```yaml
services:
  - type: web
    name: pyrosense
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    plan: free
```

### Steps

```
1. Push code to GitHub
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/USERNAME/pyrosense.git
   git push -u origin main

2. Go to render.com → New → Web Service

3. Connect GitHub repo → pyrosense

4. Settings:
   Runtime:       Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   Plan:          Free

5. Click "Create Web Service"

6. Wait ~3 minutes → app is live at:
   https://pyrosense.onrender.com
```

### Keep app awake (UptimeRobot — free)

Free Render tier sleeps after 15 minutes of inactivity.

```
1. Sign up at uptimerobot.com
2. New Monitor → HTTP(s)
3. URL: https://pyrosense.onrender.com
4. Interval: Every 14 minutes
→ App stays awake 24/7 for free ✅
```

---

## 🧰 Technologies Used

### Backend

| Technology | Role |
|------------|------|
| **Python 3.8+** | Core programming language |
| **Flask** | Lightweight web framework — handles routing, templates, HTTP |
| **scikit-learn** | Ridge Regression model + StandardScaler |
| **Pandas** | DataFrame creation (preserves feature names for scaler) |
| **NumPy** | Numerical array operations |
| **Pickle** | Serializing/loading trained model files |
| **Gunicorn** | Production WSGI server for deployment |

### Frontend

| Technology | Role |
|------------|------|
| **HTML5** | Page structure |
| **CSS3** | Styling, animations, responsive grid |
| **JavaScript (ES6+)** | Form interactivity, result display, animations |
| **Jinja2** | Template engine — renders Python variables in HTML |
| **Google Fonts** | Bebas Neue, Plus Jakarta Sans, DM Mono |
| **Canvas API** | Animated fire simulation on landing page |

### Infrastructure

| Tool | Role |
|------|------|
| **GitHub** | Version control and source code hosting |
| **Render.com** | Free cloud hosting (web service) |
| **UptimeRobot** | Keep-alive pinging (free tier) |

---

## 🔧 Troubleshooting

### ❌ `404 Not Found`
```
Problem:  Visiting home.html directly in browser
Solution: Always use /predict route — Flask serves templates,
          not static HTML files

Correct:  http://localhost:5000/predict
Wrong:    opening home.html in browser directly
```

### ❌ `500 Internal Server Error`
```
Problem:  Model files missing or dependencies not installed
Solution:
  1. Check models folder: ls models/
  2. Reinstall dependencies: pip install -r requirements.txt
  3. Check logs in terminal for exact error message
```

### ❌ Feature names mismatch error
```
Error: "The feature names should match those that were passed during fit"

Problem:  Passing numpy array to scaler instead of named DataFrame
Solution: Already fixed in app.py using:
  FEATURE_COLUMNS = list(scaler.feature_names_in_)
  df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
```

### ❌ sklearn UserWarning about feature names
```
Warning: "X does not have valid feature names, but StandardScaler
          was fitted with feature names"

Problem:  Using np.array instead of pd.DataFrame
Solution: Fixed — use pd.DataFrame with column names (see above)
Impact:   Just a warning — predictions were still correct
```

### ❌ Gateway Timeout / Page Cannot Be Displayed
```
Problem:  Institutional network/firewall blocking port 5000
Solution: 
  Option A: Use port 80 (run CMD as Administrator):
            app.run(host="0.0.0.0", port=80)
  Option B: Use port 8080:
            app.run(host="0.0.0.0", port=8080)
  Option C: Deploy to Render — accessible on standard HTTPS port 443
```

### ❌ `git push` asks for password repeatedly
```
Problem:  GitHub no longer accepts account passwords for git
Solution: Use a Personal Access Token instead:
  GitHub → Settings → Developer Settings →
  Personal Access Tokens → Tokens (classic) →
  Generate new token → check "repo" → Copy token →
  Use token as password when pushing
```

### ❌ Render build fails
```
Problem:  Missing package or wrong start command
Solution:
  1. Check requirements.txt has: flask, numpy, pandas,
     scikit-learn, gunicorn
  2. Start command must be: gunicorn app:app
     (first "app" = app.py file, second "app" = Flask instance name)
  3. Check Render logs tab for specific error
```

---

## 🔮 Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| 📍 Auto-detect region | Use IP geolocation to suggest region | Medium |
| 🌤️ Weather API | Auto-fill weather fields from OpenWeatherMap | High |
| 📈 History | Store past predictions in SQLite database | Medium |
| 🗺️ Map view | Show fire risk zones on Algeria map | Low |
| 📱 Mobile app | React Native or Flutter version | Low |
| 🔔 Alerts | Email/SMS alert when FWI exceeds threshold | High |
| 🧪 More models | Compare Random Forest, XGBoost vs Ridge | Medium |
| 📊 Dashboard | Visual analytics of prediction history | Medium |

---

## 📝 License

This project is built for **educational purposes**.

- Dataset: [Algerian Forest Fires Dataset](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) — UCI ML Repository
- FWI System: Natural Resources Canada

---

## 👤 Author

Built as part of a Machine Learning project exploring wildfire prediction and Flask web deployment.

---

<div align="center">

**⭐ If this project helped you, give it a star on GitHub!**

*Built with 🔥 Flask · Scikit-learn · Algerian Forest Fire Dataset*
BY: Nilesh Kumar Nayak
</div>
