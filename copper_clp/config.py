"""Central config — all paths and experiment constants live here."""
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_INTERIM   = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL  = ROOT / "data" / "external"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = ROOT / "reports" / "figures"
NOTEBOOKS_DIR  = ROOT / "notebooks"

for _d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL,
           MODELS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data files ─────────────────────────────────────────────────────────
COPPER_RAW  = DATA_RAW / "copper_hgf_daily.csv"    # HG=F futures from yfinance
CLPUSD_RAW  = DATA_RAW / "clpusd_daily.csv"        # CLPUSD=X from yfinance

# ── Processed ─────────────────────────────────────────────────────────────
DAILY_PANEL    = DATA_PROCESSED / "daily_panel.csv"
FEATURE_MATRIX = DATA_PROCESSED / "feature_matrix.csv"

# ── Experiment constants ───────────────────────────────────────────────────
START_DATE = "2004-01-01"

# Signal lag policy:
#   ALL copper features are shifted by at least 1 day relative to the TARGET.
#   This means we use copper data known at market close of day T-1
#   to predict CLP direction on day T.
#   We never use same-day copper to predict same-day CLP.
MIN_COPPER_LAG = 1   # minimum lag applied to every copper feature

# Walk-forward parameters
WF_TRAIN_DAYS = 252 * 3   # 3 years in-sample
WF_TEST_DAYS  = 252        # 1 year out-of-sample
WF_STEP_DAYS  = 63         # re-train every quarter

# TimesFM
TIMESFM_CONTEXT = 512
TIMESFM_HORIZON = 21
TIMESFM_REPO    = "google/timesfm-2.5-200m-pytorch"
