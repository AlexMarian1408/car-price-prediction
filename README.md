# Car Price Prediction — IMLP6 Task 04

## Scop
Proiect complet de machine learning pentru **regresie**, cu scopul de a estima prețul (`priceUSD`) unei mașini second-hand pe baza caracteristicilor sale.

## Structura proiectului

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   ├── car_price_model.joblib
│   ├── model_comparison.csv
│   └── evaluation_metrics.json
├── README.md
└── requirements.txt
```

## Setul de date
Fișierul `data/cars.csv` conține 56.244 de înregistrări inițiale și coloanele:
`make`, `model`, `priceUSD`, `year`, `condition`, `mileage(kilometers)`, `fuel_type`, `volume(cm3)`, `color`, `transmission`, `drive_unit`, `segment`.

Variabila țintă este **`priceUSD`** (redenumită intern `price_usd`).

## EDA și probleme identificate
Notebook-ul `notebooks/01_eda.ipynb` prezintă:
- dimensiunea setului, tipurile de date și valorile lipsă;
- separarea coloanelor numerice/categorice;
- distribuția țintei și valori extreme;
- relația dintre an, kilometraj, marcă și preț;
- observațiile care au ghidat curățarea și ingineria caracteristicilor.

În datele brute au fost identificate 87 de rânduri duplicate. Există valori lipsă în `volume(cm3)`, `drive_unit` și `segment`. Unele valori de kilometraj depășesc 1.000.000 km și sunt tratate ca invalide/missing. Valorile de motor de tip 16.000–20.000 cm³ sunt tratate ca erori de introducere de ordin x10 și sunt corectate prin împărțire la 10.

## Curățarea datelor
`src/data_cleaning.py`:
- normalizează denumirile coloanelor;
- standardizează textele (lowercase + trim);
- elimină duplicatele;
- validează ținta și anul;
- tratează valorile invalide de kilometraj;
- corectează valorile suspecte ale cilindreei;
- lasă valorile predictorilor lipsă pentru imputarea din pipeline.

## Ingineria caracteristicilor
`src/feature_engineering.py` creează:
- `car_age` — vârsta mașinii;
- `mileage_per_year` — kilometraj mediu pe an;
- `engine_volume_liters` — cilindree în litri;
- `is_newer_car` — indicator pentru mașini din 2015+;
- `is_high_mileage` — indicator pentru kilometraj de cel puțin 300.000 km;
- `make_model` — combinația marcă + model.

## Preprocesare
`src/data_preprocessing.py` definește:
- `SimpleImputer` pentru valori lipsă;
- `StandardScaler` pentru coloanele numerice;
- `OneHotEncoder` pentru modelul liniar;
- `OrdinalEncoder` pentru modelele bazate pe arbori/boosting;
- `ColumnTransformer` și pipeline-uri sklearn, astfel încât preprocesarea să fie învățată exclusiv din setul de antrenare.

## Modele comparate
`src/model_comparison.py` compară minimum trei modele:
1. Ridge Regression;
2. Random Forest Regressor;
3. HistGradientBoosting Regressor.

Metricile folosite sunt **MAE**, **RMSE** și **R²**, pe un set test de 20%, cu `random_state=42`.

Rezultatele rulării finale sunt:

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| HistGradientBoosting | 1.173,28 | 2.576,08 | 0,9011 |
| RandomForest | 1.111,58 | 2.939,32 | 0,8713 |
| Ridge | 2.118,84 | 4.166,66 | 0,7413 |

Random Forest are MAE ușor mai mic, însă **HistGradientBoosting** are cel mai mic RMSE și cel mai mare R²; de aceea este ales modelul final. Rezultatele sunt salvate în `models/model_comparison.csv`, iar modelul final în `models/car_price_model.joblib`.

## Cum rulezi proiectul
Din folderul proiectului:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Instalare biblioteci:
```bash
pip install -r requirements.txt
```

EDA:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

Compară cele trei modele și salvează automat modelul final:
```bash
python src/model_comparison.py
```

Evaluează modelul final salvat:
```bash
python src/model_evaluation.py
```

Alternativ, antrenarea directă a modelului final:
```bash
python src/model_training.py
```

## Interpretarea rezultatului
Un model bun trebuie să aibă MAE/RMSE cât mai mici și R² cât mai apropiat de 1. Nu se așteaptă ca fiecare preț să fie ghicit exact; scopul este reducerea erorii pe date nevăzute și documentarea clară a întregului flux: EDA → curățare → feature engineering → preprocesare → antrenare → evaluare → comparație → alegerea modelului final.
