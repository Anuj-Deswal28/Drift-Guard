# DriftGuard

**A lightweight monitoring tool that detects silent data drift and performance decay in production ML systems — and pinpoints which feature is responsible.**

Most ML monitoring only tracks system uptime and latency, not decision quality. Models silently degrade as input data drifts and user behavior shifts, often going unnoticed for weeks. DriftGuard detects feature-level data drift, tracks live performance decay, and surfaces root-cause hints — turning a generic "something broke" alert into an actionable diagnostic.

---

## The problem, demonstrated

To validate the system, a controlled drift scenario was injected into a 10-day simulated request stream: `Credit_Score`, the single most important feature to the underlying model (37% of its decision weight), was shifted down by 120 points starting on day 8.

Every layer of the system caught it, independently, and agreed:

| Signal | Before drift (days 1–7) | After drift (days 8–10) |
|---|---|---|
| PSI (`Credit_Score`) | ~0.05–0.08 (no drift) | **3.4–3.6** (severe drift; >0.25 is "significant") |
| KS test p-value | 0.2–0.97 (not significant) | **~10⁻⁵¹** (extremely significant) |
| Model accuracy | 91.5%–96% | **75.5%–82.5%** |
| Root-cause correlation | — | `Credit_Score`: **-0.955** (p = 0.0008), clearly separated from every other feature |

No other feature showed meaningful drift or correlation — the system correctly isolated the true cause rather than flagging noise.

---

## Architecture

```
Incoming request stream (features + prediction)
        ↓
  Ingestion layer — logs each request (features, prediction, timestamp) to SQLite
        ↓
  Drift Detection — sliding-window statistical comparison vs. a baseline window
        (Population Stability Index + Kolmogorov–Smirnov test, per feature)
        ↓
  Performance Tracking — once (delayed) labels arrive, computes rolling
        accuracy and AUC vs. a baseline period
        ↓
  Root-Cause Correlation — correlates each feature's drift magnitude, over time,
        against the performance decline, and ranks likely culprits
        ↓
  FastAPI backend  →  HTML + Chart.js dashboard
        (drift chart, performance trend, root-cause ranking, alert banner)
```

**Tech stack:** Python, FastAPI, SQLite, scikit-learn, SciPy/NumPy, pandas, Chart.js.

---

## How it works

### 1. Model wrapper (`model/`)
A thin adapter (`Drift_guard`) standardizes any scikit-learn/XGBoost-style model behind two methods — `.pred()` and `.pred_prob()` — with automatic fallback to raw predictions if a model doesn't support probability outputs. This is the only part of the system that touches the model directly; everything downstream works purely from logged data.

### 2. Ingestion (`Ingestion/`)
Every request is logged to SQLite as a flat row: 11 raw feature values, prediction, probability, timestamp, and a `true_label` column that starts `NULL` and is filled in once ground truth is available — modeling the delayed-label reality of production systems. Features are stored as individual columns (not a JSON blob), since per-feature drift analysis is the system's core use case.

### 3. Drift detection (`Drift/`)
Two independent, industry-standard methods, computed per feature:
- **PSI (Population Stability Index)** — percentile-based binning against a baseline window, measuring how much a feature's distribution has shifted.
- **KS test** — a statistical significance test on whether two samples plausibly come from the same distribution.

Using two methods matters: PSI gives a magnitude of shift, KS test gives a confidence level. Both were validated against synthetic known-drift/no-drift cases before being trusted on real data.

### 4. Performance tracking (`performance/`)
Rolling accuracy and AUC, computed per time window and compared against a baseline period.

### 5. Root-cause correlation (`rootcause/`)
For each feature, drift scores are computed across multiple daily windows and correlated (Pearson) against the performance metric over the same windows. The feature with the strongest, most statistically significant *negative* correlation is surfaced as the likely cause.

**A known limitation, found and confirmed on real data during development:** with only ~7 comparison points, correlation on a genuinely non-drifted feature can occasionally look misleadingly strong by chance (a "no-drift" test run showed `Loan_Amount` at correlation -0.83, p=0.02 — plausible-looking but spurious). The dashboard's alert banner only fires when correlation is both strong (< -0.5) **and** statistically significant (p < 0.05), and a larger sample of comparison windows would reduce this further. Correlation is evidence of a likely cause, not proof of one.

### 6. API + Dashboard (`api/`, `dashboard/`)
FastAPI exposes five read-only endpoints (`/drift`, `/drift/timeseries`, `/performance`, `/performance/timeseries`, `/rootcause`) that query already-logged data — DriftGuard never runs inference itself, matching how a real monitoring tool would sit alongside a production system rather than in its critical path. A single-file HTML + Chart.js dashboard consumes these endpoints.

---

## Running it

```bash
# 1. Install
pip install -e .
pip install -r requirements.txt   # fastapi, uvicorn, scikit-learn, scipy, pandas, pytest

# 2. Get the data
# Download the loan approval dataset from Kaggle and place it at:
#   data/loan_approval_data.csv

# 3. Set up the database
python -m Ingestion.setup_db          # creates the SQLite schema

# 4. Generate a simulated request stream
python -m demo.simulate_stream                    # clean, no drift
# — or —
python -m demo.simulate_stream_with_drift          # with Credit_Score drift on days 8–10

# 5. Start the API
uvicorn api.main:app --reload

# 6. Open the dashboard
# Open dashboard/index.html directly in a browser (uvicorn must be running)
```

### Running tests
```bash
pytest
```
Covers PSI, KS test, performance metrics, and root-cause correlation — each validated against hand-calculable or synthetic known-answer cases, independent of the live database.

---

## Project structure

```
driftguard/
  model/          # ModelWrapper — standardizes any sklearn-style model
  data/           # data loading, cleaning, train/test split
  Ingestion/      # SQLite schema, logging, and querying
  Drift/          # PSI + KS test drift detection
  performance/    # accuracy / AUC tracking
  rootcause/      # drift-vs-performance correlation
  api/            # FastAPI backend
  dashboard/      # HTML + Chart.js frontend
  demo/           # model training + simulated stream generation
  test/           # pytest suite
```

---

## Scope

**In scope (v1):**
- Tabular scikit-learn/XGBoost-style models
- PSI + KS-test feature drift detection
- Rolling accuracy/AUC performance tracking with simulated delayed labels
- Drift-vs-performance correlation for root-cause ranking
- A live dashboard with an alert banner

**Out of scope, future work:**
- Deep learning / image / text model support
- Automated retraining trigger recommendations
- Multi-model, multi-tenant monitoring
- Production alerting integrations (Slack, PagerDuty, email)
- Causal (not just correlational) root-cause analysis
- Dynamic feature schemas — v1's feature set and date ranges are specific to this demo's dataset and are not yet auto-detected or user-configurable
- Streaming infrastructure (Kafka), a real time-series database, and horizontal scaling for production traffic volumes

---

## Why these design choices

**Why PSI and KS test, not just watching raw accuracy?** Both catch problems *before* labels even arrive — critical in production, where ground-truth labels are often delayed by days or weeks. Waiting for accuracy to visibly drop means waiting for real, sometimes costly, downstream harm.

**Why train the model once and freeze it during simulation, instead of continuously retraining?** The entire premise of drift monitoring is that a static model's assumptions become stale as the world changes. Continuously retraining would let the model chase the data and never actually go stale — making drift undetectable by construction.

**Why held-out test data for the simulated stream, not training data?** Evaluating a model on rows it memorized during training inflates apparent performance and would distort the baseline-vs-drift performance comparison this project is built around.
