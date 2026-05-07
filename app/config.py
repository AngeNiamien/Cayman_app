from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = APP_DIR / "logo1.png"
OUTPUT_DIR = APP_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
TEMPLATE_DIR = APP_DIR / "Rapport" / "templates"

for directory in (OUTPUT_DIR, CHARTS_DIR, REPORT_DIR, TEMPLATE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

WORKFLOW = [
    "Accueil",
    "Manquants",
    "Graphiques",
    "Discriminant",
    "Modèles",
    "Performance",
    "Segmentation",
    "Rapport",
]
