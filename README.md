# 🏥 ICU Analytics System

A **production-ready local ICU monitoring + analytics + digital twin system** powered by the eICU Collaborative Research Database.

---

## 🏗️ Architecture

```
icu_system/
├── config/
│   └── config.py          ← All paths & settings
├── utils/
│   ├── data_loader.py     ← Loads .csv.gz files (chunked), caches to Parquet
│   └── db_manager.py      ← SQLite read/write helpers
├── models/
│   └── ml_model.py        ← Random Forest mortality predictor + Digital Twin
├── backend/
│   └── api.py             ← FastAPI REST backend
├── frontend/
│   └── dashboard.py       ← Streamlit dashboard UI
├── data/                  ← Auto-created: SQLite DB + Parquet cache
├── setup.py               ← Run ONCE to process data + train model
├── generate_demo_data.py  ← Generate synthetic data (no eICU needed)
├── launch.bat             ← Windows launcher menu
└── requirements.txt
```

---

## ⚡ Quick Start

### Option A — With Real eICU Data

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your data path** in `config/config.py`:
   ```python
   EICU_RAW_PATH = Path(r"D:\physionet-data\eicu\eicu-collaborative-research-database-2.0")
   ```

3. **Run setup** (once — takes 5–15 min depending on disk speed):
   ```bash
   python setup.py
   ```

4. **Start the backend** (Terminal 1):
   ```bash
   python backend/api.py
   ```

5. **Start the dashboard** (Terminal 2):
   ```bash
   streamlit run frontend/dashboard.py
   ```

6. Open browser → **http://localhost:8501**

---

### Option B — Without eICU (Demo/Test Mode)

1. Generate synthetic ICU data:
   ```bash
   python generate_demo_data.py
   ```

2. Update `config/config.py`:
   ```python
   EICU_RAW_PATH = Path("demo_data")
   ```

3. Continue from Step 3 above.

---

### Windows One-Click Launcher
Double-click **`launch.bat`** for a menu-driven launcher.

---

## 🌐 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /stats` | System-wide ICU statistics |
| `GET /patients` | List patients (with filtering) |
| `GET /patients/{id}` | Patient details |
| `GET /patients/{id}/vitals` | Vital signs time-series |
| `GET /patients/{id}/labs` | Lab results |
| `GET /patients/{id}/diagnosis` | Diagnoses |
| `GET /patients/{id}/treatments` | Treatments |
| `GET /patients/{id}/predict` | ML mortality prediction |
| `GET /patients/{id}/digital-twin` | Forecast next vitals |
| `GET /analytics/unit-breakdown` | Stats by ICU unit |
| `GET /analytics/mortality-by-age` | Mortality by age group |
| `GET /analytics/top-diagnoses` | Most common diagnoses |

**Interactive API docs:** http://127.0.0.1:8000/docs

---

## 🧠 ML Model

- **Algorithm:** Random Forest Classifier (100 trees, depth 8)
- **Target:** Hospital mortality (binary)
- **Features:** Age, APACHE score, mean vitals (HR, SpO2, BP, temp, resp), lab values
- **Output:** Risk score (0–1), risk label (LOW / MODERATE / HIGH), top predictive features
- **Storage:** `models/mortality_model.pkl` — trained once, reloaded on every run

---

## 📡 Digital Twin

The digital twin forecasts the next N vital sign values (default: 6 steps × 5 min = 30 min) using:
- **Exponential Weighted Moving Average** for base value
- **Linear trend component** (weighted at 30%)
- **Confidence bands** based on recent observation variance

---

## ⚙️ Performance Design

| Problem | Solution |
|---------|----------|
| Large .csv.gz files | Chunked reading (50k rows/chunk) |
| Slow re-processing | Parquet cache (read once, reuse forever) |
| Heavy memory usage | Only load needed columns |
| Slow API queries | SQLite with indexed patientunitstayid |
| Model retraining | Pickled model, load on startup |

---

## 🔧 Configuration

Edit `config/config.py` to change:
- `EICU_RAW_PATH` — where your .csv.gz files live
- `CHUNK_SIZE` — rows per chunk (default 50,000)
- `VITALS_SAMPLE_ROWS` — how many vitalPeriodic rows to load (default 500,000)
- `API_PORT` — backend port (default 8000)

---

## 🔒 Future Security Extensions

The codebase is structured for easy extension:
- Add JWT authentication to `backend/api.py`
- Add HTTPS via `uvicorn --ssl-keyfile/certfile`
- Add role-based access (doctor/nurse/admin) as middleware
- Replace SQLite with PostgreSQL for multi-user access

---

## 📊 Dashboard Pages

| Page | Features |
|------|----------|
| **Overview** | System stats, unit breakdown, mortality by age, top diagnoses |
| **Patient Search** | Filter by ID/unit/risk, sortable table |
| **Patient Detail** | Vitals charts, lab trends, diagnosis list, ML prediction gauge |
| **Analytics** | Population-level charts, APACHE vs mortality |
| **Digital Twin** | Vital forecasting with confidence bands |

---

## 🐛 Troubleshooting

**"Cannot connect to backend"**
→ Start `python backend/api.py` first

**"Patient not found"**
→ Run `python setup.py` to populate the database

**"ML model not trained"**
→ Run `python setup.py` — it trains automatically

**Setup is slow**
→ Normal for first run. Subsequent runs use Parquet cache (instant).

**Out of memory**
→ Reduce `VITALS_SAMPLE_ROWS` in `config/config.py`
