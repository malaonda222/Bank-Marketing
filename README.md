# bank-marketing-classification

[Italiano](#italiano) | [English](#english)

---

## Italiano

Progetto di studio realizzato per consolidare competenze in Python e Machine Learning — parte di un percorso di formazione come Data Analyst.

Analisi end-to-end sul dataset **Bank Marketing UCI** (campagna di telemarketing di una banca portoghese, 2008-2010), con pipeline completa che parte dalla pulizia del dato grezzo fino alla modellazione predittiva in sklearn e alla replica visuale in Dataiku DSS.

---

### Struttura del progetto

```
Bank_Marketing/
│
├── raw_dataset/
│   └── bank-additional-full.csv        # Dataset originale UCI (41188 righe × 21 colonne)
│
├── cleaned_dataset/
│   └── bank_marketing_cleaned.csv      # Dataset pulito, output della fase di cleaning
│
├── cleaning_dataset/
│   └── cleaning_bank_marketing.ipynb   # Notebook di pulizia dati
│
├── eda_classificazione/
│   ├── eda_classificazione_bank_marketing.ipynb  # Notebook EDA + modellazione
│   └── plots/                          # Grafici salvati durante l'analisi
│
└── dataiku_flow/                       # Progetto Dataiku DSS
```

---

### Stack tecnologico

| Strumento | Utilizzo |
|---|---|
| **Python 3** | Pulizia dati, EDA, modellazione |
| **pandas** | Manipolazione e trasformazione dati |
| **matplotlib / seaborn** | Visualizzazioni statistiche |
| **scikit-learn** | Classificazione binaria, pipeline, metriche |
| **Dataiku DSS** | Replica visuale del flow di modellazione |
| **Jupyter Notebook** | Ambiente di sviluppo principale |
| **PostgreSQL** (Docker) | Ambiente di supporto al percorso formativo |

---

### Il problema: classificazione binaria sbilanciata

Il dataset contiene i dati di una campagna di telemarketing bancario. L'obiettivo è predire se un cliente sottoscriverà un deposito a termine (`y = yes/no`).

**Caratteristica critica del dataset:** il target è fortemente sbilanciato — ~88% `no` / ~12% `yes`. Questo rende l'accuracy una metrica ingannevole: un modello che predice sempre `no` raggiunge 88% di accuracy senza aver imparato nulla. Le metriche corrette per questo problema sono **Precision, Recall, F1 e AUC-ROC sulla classe `yes`**.

---

### Parte 1 — Data Cleaning

**Obiettivo:** trasformare il dataset grezzo UCI in un dataset pulito, coerente e pronto per l'analisi.

**Notebook:** `cleaning_bank_marketing.ipynb`

#### Operazioni eseguite

**Caricamento e ispezione iniziale:**
- Caricamento con `sep=";"` (separatore del formato UCI)
- Ispezione shape, tipi, statistiche descrittive, distribuzione target

**Gestione valori unknown:**
- Colonne con unknown: `job`, `marital`, `education`, `default`, `housing`, `loan`
- Pipeline di imputazione: `unknown` → `NaN` → `OrdinalEncoder` temporaneo → `KNNImputer (k=5)` → `inverse_transform`
- Nessun unknown residuo dopo il processo

**Gestione di `pdays`:**
- Il valore sentinella `999` indica "mai contattato in precedenza" — non è un valore numerico reale
- Creata variabile binaria `contacted_before` (0 = mai contattato, 1 = contattato in precedenza)
- Droppata la colonna `pdays` originale

**Gestione di `previous`:**
- Lasciato invariato — `0` è un valore reale corretto (zero contatti precedenti)

**Verifica coerenze:**
- Identificate e documentate incoerenze tra `pdays` e `previous`:
  - Tipo 1: `pdays=999` ma `previous>0` (mai contattato ma con contatti registrati)
  - Tipo 2: `pdays≠999` ma `previous=0` (contattato ma zero contatti registrati)

**Conversione target:**
- `yes → 1`, `no → 0`

**Output:** `bank_marketing_cleaned.csv` (41188 righe × 22 colonne)

**Nota metodologica:** tutto il codice distingue esplicitamente tra operazioni *diagnostiche* (identificazione anomalie) e *correttive* (modifica del DataFrame).

---

### Parte 2 — EDA orientata alla classificazione

**Obiettivo:** esplorare il dataset in modo sistematico con focus sul potere discriminante delle feature rispetto al target.

**Notebook:** `eda_classificazione_bank_marketing.ipynb`

#### Analisi eseguite

**Step 2.1 — Distribuzione del target:**
- Confermato sbilanciamento 88.7% `no` / 11.3% `yes`
- Visualizzazione conteggio assoluto e percentuale

**Step 2.2 — Distribuzioni numeriche per classe:**
- Istogrammi sovrapposti per densità per tutte le variabili numeriche (`age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)
- Confronto visivo delle distribuzioni tra classe `yes` e `no`

**Step 2.3 — Tasso di sottoscrizione per variabili categoriche:**
- Tasso `yes%` per ogni categoria di `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`
- Linea di riferimento con la media globale (11.3%)
- Segnali emersi: `poutcome=success` ha tasso >60%, `contact=cellular` superiore a `telephone`, mesi `mar/dec/sep/oct` con tassi elevati

**Step 2.4 — Matrice di correlazione:**
- Correlazioni tra variabili numeriche + target
- Segnali emersi: `duration` ha la correlazione più alta con `y` (leakage), le 5 variabili macroeconomiche sono fortemente correlate tra loro (multicollinearità)

**Step 2.5 — Boxplot duration e emp.var.rate per classe:**
- Confronto quartili tra le due classi su `duration` ed `emp.var.rate`
- `duration`: box nettamente separato tra `yes` e `no` — conferma visiva del leakage
- `emp.var.rate`: separazione tra classi presente — le campagne condotte in periodi di contrazione economica hanno tassi di sottoscrizione più alti, segnale predittivo reale senza leakage

---

### Parte 3 — Modellazione sklearn

**Obiettivo:** costruire una pipeline di classificazione binaria con tre modelli a complessità crescente, gestendo sbilanciamento e leakage in modo esplicito.

#### Decisioni progettuali

| Decisione | Scelta | Motivazione |
|---|---|---|
| `duration` | Due versioni — con e senza | Leakage indiretto: nota solo dopo la chiamata |
| Macro economiche | Solo `euribor3m` | Le altre 4 sono fortemente correlate, `euribor3m` è la più rappresentativa |
| `pdays` | Esclusa | Valore sentinella 999, informazione già in `contacted_before` |
| `previous` | Esclusa | Informazione già sintetizzata in `contacted_before` |
| `contacted_before` | Inclusa | Sintesi pulita e corretta dell'informazione di contatto precedente |
| Sbilanciamento | `class_weight='balanced'` | Pesa gli errori inversamente alla frequenza della classe |

#### Preprocessing (ColumnTransformer)

| Colonne | Trasformazione |
|---|---|
| `age`, `campaign`, `euribor3m` | StandardScaler |
| `contacted_before` | Passthrough (già 0/1) |
| `education` | OrdinalEncoder con gerarchia esplicita: `illiterate < basic.4y < basic.6y < basic.9y < high.school < professional.course < university.degree` |
| `default`, `housing`, `loan` | OrdinalEncoder (binarie, compatto) |
| `job`, `marital`, `contact`, `month`, `day_of_week`, `poutcome` | OneHotEncoder (drop="first") |
| `duration` | StandardScaler, solo nel modello con leakage |

#### Baseline — DummyClassifier

| Strategia | Accuracy | F1 (yes) | Recall (yes) | AUC-ROC |
|---|---|---|---|---|
| most_frequent | 0.89 | 0.00 | 0.00 | 0.500 |
| stratified | 0.80 | 0.12 | 0.12 | 0.505 |

**Lettura:** `most_frequent` predice sempre `no` e raggiunge 89% di accuracy — conferma empirica che l'accuracy è una metrica inutile con dataset sbilanciato. Qualsiasi modello reale deve battere questi valori.

#### Risultati sklearn — versione SENZA duration (realistico)

| Modello | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.32 | 0.68 | 0.43 | 0.798 |
| Decision Tree | 0.84 | 0.37 | 0.63 | 0.46 | 0.795 |
| Random Forest | 0.86 | 0.41 | 0.64 | 0.50 | 0.811 |

#### Risultati sklearn — versione CON duration (leakage documentato)

| Modello | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.45 | 0.90 | 0.60 | 0.942 |
| Decision Tree | 0.84 | 0.41 | 0.93 | 0.57 | 0.940 |
| Random Forest | 0.87 | 0.46 | 0.93 | 0.61 | 0.948 |

**Osservazione critica documentata:** il gap di ~13-15 punti di AUC-ROC tra la versione con e senza `duration` quantifica l'effetto del leakage. `duration` è nota solo a chiamata conclusa — includerla in un modello deployabile in produzione significherebbe usare informazione non disponibile al momento della previsione. La versione realistica (senza `duration`) è l'unica deployabile.

---

### Parte 4 — Replica in Dataiku DSS

**Obiettivo:** replicare la pipeline sklearn in Dataiku DSS e confrontare le metriche tra i due ambienti.

#### Flow costruito
```
bank_marketing_cleaned (upload)
       │
       ▼
   [Prepare recipe — drop colonne escluse]
bank_marketing_cleaned_prepared
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
   [Prepare recipe]                    [Predict y (binary)]
bank_marketing_cleaned_                        │
prepared_senza_duration                        ▼
       │                          bank_marketing_cleaned_
       ▼                              prepared_scored
   [Predict y (binary)]
       │
       ▼
bank_marketing_cleaned_
prepared_senza_duration_scored
```

#### Prepare Recipe

Colonne droppate: `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`

#### Configurazione Lab

- **Target:** `y`
- **Split:** 80/20, random seed 42, Sampling method: Random
- **Sbilanciamento:** Weighting strategy → Class weights (equivalente a `class_weight='balanced'`)
- **Encoding:** replicato esattamente come in sklearn (OrdinalEncoder per `education`, `default`, `housing`, `loan` — OneHot per le nominali — StandardScaler per le numeriche)

#### Confronto metriche AUC-ROC: sklearn vs Dataiku

**Senza duration (realistico):**

| Modello | sklearn | Dataiku | Differenza |
|---|---|---|---|
| Logistic Regression | 0.798 | 0.798 | 0.000 |
| Decision Tree | 0.795 | 0.804 | +0.009 |
| Random Forest | 0.811 | 0.812 | +0.001 |

**Con duration (leakage documentato):**

| Modello | sklearn | Dataiku | Differenza |
|---|---|---|---|
| Logistic Regression | 0.942 | 0.935 | -0.007 |
| Decision Tree | 0.940 | 0.937 | -0.003 |
| Random Forest | 0.948 | 0.946 | -0.002 |

**Le piccole differenze sono attese e spiegabili:**
- Dataiku applica automaticamente un hyperparameter search (search size = 5) che può selezionare parametri diversi dai default sklearn
- La Logistic Regression con duration mostra la differenza più grande (0.942 vs 0.935) — Dataiku ha scelto `C=100` tramite search, diverso dal default `C=1` di sklearn
- Il pattern del leakage è confermato identicamente in entrambi gli ambienti: calo di ~13-15 punti AUC rimuovendo `duration`

---

### Note metodologiche

- La distinzione tra **codice diagnostico** e **codice correttivo** è mantenuta esplicita in tutti i notebook
- Le convenzioni di output in Jupyter: `display()` per DataFrame, `print()` per scalari
- `class_weight='balanced'` è applicato a tutti i modelli reali — mai omesso con dataset sbilanciati
- Il leakage di `duration` è stato identificato nell'EDA (Step 2.5), documentato nelle decisioni progettuali e quantificato nel confronto finale (+13-15 punti AUC)
- Il `random_state=42` e il `stratify=y` nello split garantiscono la proporzione 88/12 in entrambi i set e la riproducibilità dei risultati

---

### Come riprodurre il progetto

#### Prerequisiti

- Python 3.9+ con le librerie: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Jupyter Notebook
- Dataiku DSS (opzionale, per la Parte 4)

#### Esecuzione

1. Clona la repository
2. Esegui prima il notebook `cleaning_dataset/cleaning_bank_marketing.ipynb`
3. Il file `bank_marketing_cleaned.csv` viene salvato in `cleaned_dataset/`
4. Esegui poi `eda_classificazione/eda_classificazione_bank_marketing.ipynb`
5. (Opzionale) Importa il progetto in Dataiku DSS per esplorare il flow visuale

---

### Autore

Lisa Bandinelli, in formazione come Data Analyst — percorso di studio su SQL, Python, Machine Learning e strumenti enterprise come Dataiku e n8n.

---
---

<a name="english"></a>
## English

A study project built to consolidate skills in Python and Machine Learning — part of an ongoing training path as a Data Analyst.

End-to-end analysis on the **Bank Marketing UCI** dataset (telephone marketing campaign of a Portuguese bank, 2008-2010), covering a full pipeline from raw data cleaning to binary classification modelling in sklearn and visual replication in Dataiku DSS.

---

### Project Structure

```
Bank_Marketing/
│
├── raw_dataset/
│   └── bank-additional-full.csv        # Original UCI dataset (41188 rows × 21 columns)
│
├── cleaned_dataset/
│   └── bank_marketing_cleaned.csv      # Cleaned dataset, output of the cleaning phase
│
├── cleaning_dataset/
│   └── cleaning_bank_marketing.ipynb   # Data cleaning notebook
│
├── eda_classificazione/
│   ├── eda_classificazione_bank_marketing.ipynb  # EDA + modelling notebook
│   └── plots/                          # Plots saved during analysis
│
└── dataiku_flow/                       # Dataiku DSS project
```

---

### Tech Stack

| Tool | Usage |
|---|---|
| **Python 3** | Data cleaning, EDA, modelling |
| **pandas** | Data manipulation and transformation |
| **matplotlib / seaborn** | Statistical visualisations |
| **scikit-learn** | Binary classification, pipelines, metrics |
| **Dataiku DSS** | Visual replication of the modelling pipeline |
| **Jupyter Notebook** | Primary development environment |
| **PostgreSQL** (Docker) | Supporting environment for the training path |

---

### The Problem: Imbalanced Binary Classification

The dataset contains data from a bank telemarketing campaign. The goal is to predict whether a customer will subscribe to a term deposit (`y = yes/no`).

**Critical dataset characteristic:** the target is heavily imbalanced — ~88% `no` / ~12% `yes`. This makes accuracy a misleading metric: a model that always predicts `no` achieves 88% accuracy without learning anything. The correct metrics for this problem are **Precision, Recall, F1 and AUC-ROC on the `yes` class**.

---

### Part 1 — Data Cleaning

**Goal:** transform the raw UCI dataset into a clean, consistent, analysis-ready dataset.

**Notebook:** `cleaning_bank_marketing.ipynb`

#### Operations performed

**Loading and initial inspection:**
- Loaded with `sep=";"` (UCI format separator)
- Inspection of shape, dtypes, descriptive statistics, target distribution

**Handling unknown values:**
- Columns with unknown: `job`, `marital`, `education`, `default`, `housing`, `loan`
- Imputation pipeline: `unknown` → `NaN` → temporary `OrdinalEncoder` → `KNNImputer (k=5)` → `inverse_transform`
- No residual unknowns after the process

**Handling `pdays`:**
- Sentinel value `999` means "never previously contacted" — not a real numeric value
- Created binary variable `contacted_before` (0 = never contacted, 1 = previously contacted)
- Dropped original `pdays` column

**Handling `previous`:**
- Left unchanged — `0` is a real correct value (zero previous contacts)

**Consistency checks:**
- Identified and documented inconsistencies between `pdays` and `previous`:
  - Type 1: `pdays=999` but `previous>0`
  - Type 2: `pdays≠999` but `previous=0`

**Target encoding:**
- `yes → 1`, `no → 0`

**Output:** `bank_marketing_cleaned.csv` (41188 rows × 22 columns)

**Methodological note:** the code explicitly distinguishes between *diagnostic* operations (anomaly detection) and *corrective* operations (DataFrame modifications).

---

### Part 2 — Classification-Oriented EDA

**Goal:** systematically explore the dataset with focus on the discriminative power of features with respect to the target.

**Notebook:** `eda_classificazione_bank_marketing.ipynb`

**Step 2.1 — Target distribution:** confirmed 88.7% `no` / 11.3% `yes` imbalance.

**Step 2.2 — Numerical distributions by class:** overlapping density histograms for all numerical variables, comparing `yes` vs `no` distributions.

**Step 2.3 — Subscription rate by categorical variable:** `yes%` rate for each category — key signals: `poutcome=success` >60% rate, `contact=cellular` outperforms `telephone`, March/December/September/October show elevated rates.

**Step 2.4 — Correlation matrix:** `duration` has the highest correlation with `y` (leakage signal), the 5 macroeconomic variables are strongly correlated with each other (multicollinearity).

**Step 2.5 — Boxplots for duration and emp.var.rate:** clear box separation for `duration` between classes confirms leakage visually; `emp.var.rate` separation confirms genuine macroeconomic predictive signal without leakage.

---

### Part 3 — sklearn Modelling

**Goal:** build a binary classification pipeline with three models of increasing complexity, explicitly handling imbalance and leakage.

#### Key design decisions

| Decision | Choice | Motivation |
|---|---|---|
| `duration` | Two versions — with and without | Indirect leakage: known only after the call ends |
| Macroeconomic variables | Only `euribor3m` | The other 4 are redundant due to multicollinearity |
| `pdays` | Excluded | Sentinel value 999, information already in `contacted_before` |
| `previous` | Excluded | Information already synthesised in `contacted_before` |
| `contacted_before` | Included | Clean and correct synthesis of previous contact information |
| Class imbalance | `class_weight='balanced'` | Weights errors inversely proportional to class frequency |

#### Preprocessing (ColumnTransformer)

| Columns | Transformation |
|---|---|
| `age`, `campaign`, `euribor3m` | StandardScaler |
| `contacted_before` | Passthrough (already 0/1) |
| `education` | OrdinalEncoder with explicit hierarchy: `illiterate < basic.4y < basic.6y < basic.9y < high.school < professional.course < university.degree` |
| `default`, `housing`, `loan` | OrdinalEncoder (binary, compact) |
| `job`, `marital`, `contact`, `month`, `day_of_week`, `poutcome` | OneHotEncoder (drop="first") |
| `duration` | StandardScaler, only in the leakage model |

#### Baseline — DummyClassifier

| Strategy | Accuracy | F1 (yes) | Recall (yes) | AUC-ROC |
|---|---|---|---|---|
| most_frequent | 0.89 | 0.00 | 0.00 | 0.500 |
| stratified | 0.80 | 0.12 | 0.12 | 0.505 |

#### Results — WITHOUT duration (realistic)

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.32 | 0.68 | 0.43 | 0.798 |
| Decision Tree | 0.84 | 0.37 | 0.63 | 0.46 | 0.795 |
| Random Forest | 0.86 | 0.41 | 0.64 | 0.50 | 0.811 |

#### Results — WITH duration (leakage documented)

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.45 | 0.90 | 0.60 | 0.942 |
| Decision Tree | 0.84 | 0.41 | 0.93 | 0.57 | 0.940 |
| Random Forest | 0.87 | 0.46 | 0.93 | 0.61 | 0.948 |

**Documented critical observation:** the ~13-15 point AUC-ROC gap between the two versions quantifies the leakage effect. `duration` is only known after the call ends — including it in a production model would mean using information unavailable at prediction time. The realistic version (without `duration`) is the only deployable one.

---

### Part 4 — Replication in Dataiku DSS

**Goal:** replicate the sklearn pipeline in Dataiku DSS and compare metrics across both environments.

#### Built Flow

```
bank_marketing_cleaned (upload)
       │
       ▼
   [Prepare recipe — drop colonne escluse]
bank_marketing_cleaned_prepared
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
   [Prepare recipe]                    [Predict y (binary)]
bank_marketing_cleaned_                        │
prepared_senza_duration                        ▼
       │                          bank_marketing_cleaned_
       ▼                              prepared_scored
   [Predict y (binary)]
       │
       ▼
bank_marketing_cleaned_
prepared_senza_duration_scored
```

#### AUC-ROC Comparison: sklearn vs Dataiku

**Without duration (realistic):**

| Model | sklearn | Dataiku | Difference |
|---|---|---|---|
| Logistic Regression | 0.798 | 0.798 | 0.000 |
| Decision Tree | 0.795 | 0.804 | +0.009 |
| Random Forest | 0.811 | 0.812 | +0.001 |

**With duration (leakage documented):**

| Model | sklearn | Dataiku | Difference |
|---|---|---|---|
| Logistic Regression | 0.942 | 0.935 | -0.007 |
| Decision Tree | 0.940 | 0.937 | -0.003 |
| Random Forest | 0.948 | 0.946 | -0.002 |

Results are substantially aligned across both environments. Small differences are explained by Dataiku's automatic hyperparameter search. The leakage pattern is confirmed identically in both tools.

---

### Methodological Notes

- The distinction between **diagnostic** and **corrective** code is kept explicit throughout all notebooks
- Jupyter output conventions: `display()` for DataFrames, `print()` for scalar values
- `class_weight='balanced'` is applied to all real models — never omitted with imbalanced datasets
- The leakage of `duration` was identified in the EDA (Step 2.5), documented in design decisions, and quantified in the final comparison (+13-15 AUC points)
- `random_state=42` and `stratify=y` in the split guarantee the 88/12 proportion in both sets and full reproducibility

---

### How to Reproduce

#### Prerequisites

- Python 3.9+ with: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Jupyter Notebook
- Dataiku DSS (optional, for Part 4)

#### Steps

1. Clone the repository
2. Run `cleaning_dataset/cleaning_bank_marketing.ipynb` first
3. `bank_marketing_cleaned.csv` will be saved in `cleaned_dataset/`
4. Then run `eda_classificazione/eda_classificazione_bank_marketing.ipynb`
5. (Optional) Import the project into Dataiku DSS to explore the visual flow

---

### Author

Lisa Bandinelli, aspiring Data Analyst — training path covering SQL, Python, Machine Learning, and enterprise tools such as Dataiku and n8n.
