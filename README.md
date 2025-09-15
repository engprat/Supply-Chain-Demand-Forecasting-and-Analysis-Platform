# Supply-Chain-Demand-Forecasting-and-Analysis-Platform
This is a modular demand-forecasting backend that predicts daily customer orders at customer × product × city granularity. It combines internal history with external context (e.g., promotions, inventory days, sell-outs) and returns both row-level predictions and summary statistics for analytics, planning, and downstream BI.

---

## Quick Start

### Requirements
- **Python** 3.10+
- **Node** 18+ (for the UI)
- (Optional) **Git LFS** for model/report artifacts

## 1) Backend (FastAPI)
```bash
# From repo root
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r src/backend/requirements.txt
uvicorn src.backend.app.main:app --reload --port 8000

# Backend: How Predictions Are Built

1. **Ingestion & enrichment**: Load orders; join external/internal signals (promotions, inventory days, sell-outs).
2. **Feature engineering**: Lags, moving stats, seasonality flags, promo features, inventory constraints.
3. **Modeling**: Time-series (e.g., classical/stat ML) + regression/GBM features for promo/constraints.
4. **Serving**: FastAPI endpoints compute predictions on request; stats (min/median/p90/p95/max/mean/std) included.
5. **Export & caching**: CSV/JSON responses, cache keys returned for traceability.

### Ops & config
- **Env**: `BACKEND_URL`, model paths, data locations.
- **Performance**: vectorized ops, paging, percentile filters (`min_pct`), quantity gates (`min_qty`).
- **Security**: enable CORS rules and auth (add before production), input validation on all POST bodies.
- **Testing**: unit tests per feature module; contract tests for API schemas.

---

```
## 2) Frontend
``` Global capabilities (project-wide)
- **Layout & Navigation**
  - Side nav switches windows (`dashboard`, `simulation`, `promotional`, `adhoc-forecast`, `aggregation`, `customerOrders`).
  - Connection widget to check backend status; global error banners.
- **Data export**
  - Header “Export Data” (CSV/JSON/XML) serializes the active window’s data.
- **Charts & UX**
  - Responsive charts, tooltips, legends, animated first render; accessible controls and loading states.

```
## 3) Project Objective
Provide accurate, explainable demand forecasts and customer-level order predictions that planners can filter, simulate, and export. The system combines internal history (orders, inventory, sell-outs) with external/contextual signals (e.g., promotions, calendar effects) and serves results through a FastAPI backend and a React frontend.

## 4) Problem Statement
Traditional forecasting often stops at SKU or channel totals and struggles to incorporate promo effects, stockouts, or fast-changing signals. Planners need daily, customer-level forecasts with:
- Controls to filter by customer/product/city and apply thresholds (min qty / percentile)
  
- Separation of base demand vs. promo-lift
  
- Aggregations/drill-downs across hierarchy levels
  
- Scenario testing for “what-if” changes
  
- All of this must be automated, reproducible, and API-driven for downstream use.

## 5) Data & Inputs

- Historical orders (date, customer_id, product_id, city, qty)

- Promotions (flags/intensity/timing)

- Inventory days / sell-outs (to interpret suppressed demand)

- Calendar signals (dow/holiday/seasonality)

- Optional channel context (e-commerce, etc.)

## 6) Preprocessing

- Schema alignment & type coercion: ensure consistent keys and datatypes.

- De-duplication & range checks: remove duplicates, enforce valid dates/ids.

- Outlier handling: cap/extreme smoothing to reduce distortion.

- Missing data & stockout logic: impute small gaps; treat sell-outs as constrained supply (avoid learning “zeros” as true demand).

- Promo calendar join: align promo periods to daily grain with lead/lag windows.

- Train/validation split (time-aware): no leakage across time boundaries.

## 7) Model & Training

- Approach: Blend of classical time-series (for stable seasonality/trend) and feature-driven ML (for promo/interaction effects).
Exact algorithms are modular—can be ARIMA/ETS/regression/GBM; the service abstracts this behind endpoints.

- Targets: daily demand / customer-level orders.

- Loss & metrics: MAE/RMSE/MAPE tracked by segment; percentile stats (median/p90/p95) produced for UI context.

- Validation: rolling/time-based validation; backtests to estimate stability.

- Versioning: model artifacts keyed for traceability (cache_key surfaced by API).

## 8) Important Operational Aspects

- Config & env: endpoints, model paths, data locations; CORS set for local dev.

- Monitoring: log latency, error rates, and segment MAPE; track drift on key features.

- Reproducibility: pinned data snapshots and model versions; deterministic pipelines where possible.

- Extensibility: drop-in new signals (price, weather), add quantiles/intervals, add auth/rate-limits for production.

## 9) Project Structure And Workflow

```text
├── LICENSE
├── README.md
├── notebooks
│   └── store_demand.py
├── reports
│   └── Report.pdf
├── src
│   ├── backend
│   │   ├── app
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── routers
│   │   │   │   ├── data_summary.py
│   │   │   │   ├── forecast.py                # /api/forecast, /api/daily-forecast, /api/scenario-forecast
│   │   │   │   ├── promotional.py             # /api/promotional-forecast
│   │   │   │   ├── aggregation.py             # /api/aggregate-forecast, /api/drill-down
│   │   │   │   └── customer_orders.py         # /api/customer-orders/predict, /options
│   │   │   ├── services
│   │   │   │   ├── preprocessing.py
│   │   │   │   ├── features.py
│   │   │   │   ├── models.py                  # load/serve model artifacts
│   │   │   │   ├── inference.py               # core predict logic
│   │   │   │   └── stats.py                   # p50/p90/p95, summaries
│   │   │   ├── data
│   │   │   │   ├── raw/                       # raw snapshots (if bundled)
│   │   │   │   └── processed/                 # cached/cleaned subsets
│   │   │   ├── artifacts
│   │   │   │   └── models/                    # serialized model(s), scalers, encoders
│   │   │   └── utils
│   │   │       ├── io.py                      # CSV/JSON export helpers
│   │   │       ├── cache.py
│   │   │       └── schemas.py                 # pydantic request/response schemas
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend
│       └── forecast_ui
│           ├── package.json
│           └── src
│               ├── index.js
│               ├── index.css
│               ├── App.js
│               ├── App.css
│               ├── components
│               │   ├── Layout
│               │   │   ├── Layout.js
│               │   │   └── Layout.css
│               │   ├── Charts
│               │   │   ├── Charts.css
│               │   │   ├── DailyForecastChart.js
│               │   │   ├── PromotionalAnalysisChart.js
│               │   │   ├── SimpleForecastChart.js
│               │   │   └── ForecastChart.js
│               │   ├── ConnectionTest
│               │   │   ├── ConnectionTest.js
│               │   │   └── ConnectionTest.css
│               │   ├── ControlPanel
│               │   │   ├── DailyForecastControl.js
│               │   │   ├── DailyForecastControl.css
│               │   │   ├── PromotionalAnalysisControl.js
│               │   │   ├── PromotionalAnalysisControl.css
│               │   │   ├── AdHocForecastControl.js
│               │   │   └── AdHocForecastControl.css
│               │   ├── ScenarioPanel
│               │   │   ├── ScenarioPanel.js
│               │   │   └── ScenarioPanel.css
│               │   └── views
│               │       ├── DashboardWindow
│               │       │   ├── DashboardWindow.js
│               │       │   └── DashboardWindow.css
│               │       ├── SimulationWindow
│               │       │   ├── SimulationWindow.js
│               │       │   └── SimulationWindow.css
│               │       ├── PromotionalWindow
│               │       │   ├── PromotionalWindow.js
│               │       │   └── PromotionalWindow.css
│               │       ├── AdHocWindow
│               │       │   ├── AdHocWindow.js
│               │       │   └── AdHocWindow.css
│               │       ├── AggregationWindow
│               │       │   ├── AggregationWindow.js
│               │       │   └── AggregationWindow.css
│               │       └── CustomerOrdersWindow
│               │           ├── CustomerOrdersVizPanel.js
│               │           ├── CustomerOrdersVizPanel.css
│               │           └── CustomerOrdersPies.js
│               └── assets/                     # optional
└── scripts/                                    # optional (data pulls, batch jobs)
