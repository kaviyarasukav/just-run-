import os
import zipfile
import pandas as pd

OUT_DIR = "./output"
CSV_DIR = "./output/csv"
XLSX_PATH = "./output/optimization_report.xlsx"
ZIP_PATH = "./output/quant_analysis_bundle.zip"

def export_csvs():
    os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.exists(XLSX_PATH):
        print(f"[ERR] {XLSX_PATH} not found.")
        return []
    
    xl = pd.ExcelFile(XLSX_PATH)
    exported_files = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        clean_name = sheet.replace(" ", "_").replace("/", "_")
        csv_filename = f"{CSV_DIR}/{clean_name}.csv"
        df.to_csv(csv_filename, index=False)
        exported_files.append(csv_filename)
    print(f"[csv] Exported {len(exported_files)} sheets to {CSV_DIR}/")
    return exported_files

def create_executive_html():
    exec_html_path = f"{OUT_DIR}/executive_report.html"
    summary_csv = f"{CSV_DIR}/Summary.csv"
    data_csv = f"{CSV_DIR}/Data_Features.csv"
    doc_csv = f"{CSV_DIR}/Formulas_Documentation.csv"
    
    summary_df = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    data_df = pd.read_csv(data_csv) if os.path.exists(data_csv) else pd.DataFrame()
    doc_df = pd.read_csv(doc_csv) if os.path.exists(doc_csv) else pd.DataFrame()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Strategy & Data Optimization Master Executive Report</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; line-height: 1.6; margin: 0; padding: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; }}
        .header h1 {{ color: #0f172a; margin: 0 0 10px 0; font-size: 2.2rem; }}
        .header p {{ color: #64748b; margin: 0; font-size: 1.1rem; }}
        .section {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 25px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .section-title {{ font-size: 1.4rem; color: #1e3a8a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; margin-top: 0; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9rem; }}
        th {{ background-color: #1e293b; color: #ffffff; text-align: left; padding: 10px; font-weight: 600; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
        tr:nth-child(even) {{ background-color: #f1f5f9; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #cbd5e1; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; background: #e0f2fe; color: #0369a1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Quantitative Strategy & Data Optimization Executive Report</h1>
        <p>Exponential Moving Average Stop-And-Reverse (EMA SAR) Multi-Asset Universe Evaluation</p>
    </div>

    <div class="section">
        <h2 class="section-title">📐 Mathematical Formulas & Methodology Documentation</h2>
        {doc_df.to_html(classes="table", index=False) if not doc_df.empty else "<p>No doc data</p>"}
    </div>

    <div class="section">
        <h2 class="section-title">🏆 Top Universe Strategy Summary (Ranked by Profit %)</h2>
        {summary_df.head(15).to_html(classes="table", index=False) if not summary_df.empty else "<p>No summary data</p>"}
    </div>

    <div class="section">
        <h2 class="section-title">🖼️ Portfolio Universe Visual Analytics</h2>
        <div class="grid-2">
            <div>
                <h3>Universe Return vs Risk Frontier</h3>
                <img src="portfolio_risk_return.png" alt="Risk Return Frontier">
            </div>
            <div>
                <h3>Profit Factor Matrix (Symbol vs Timeframe)</h3>
                <img src="universe_profit_matrix.png" alt="Universe Matrix">
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">📊 Dataset Properties & Metadata (Before Testing)</h2>
        {data_df.to_html(classes="table", index=False) if not data_df.empty else "<p>No data profiles</p>"}
    </div>
</body>
</html>
"""
    with open(exec_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] Executive HTML report generated -> {exec_html_path}")

def build_zip_package():
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUT_DIR):
            for file in files:
                if file.endswith(".zip"):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUT_DIR)
                zipf.write(file_path, arcname)
    print(f"[zip] Bundle ZIP created -> {ZIP_PATH}")

if __name__ == "__main__":
    export_csvs()
    create_executive_html()
    build_zip_package()
