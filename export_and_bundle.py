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
    
    charts_csv = f"{CSV_DIR}/Charts.csv"
    if os.path.exists(charts_csv):
        try: os.remove(charts_csv)
        except Exception: pass

    xl = pd.ExcelFile(XLSX_PATH)
    exported_files = []
    for sheet in xl.sheet_names:
        if sheet.strip().lower() in ("charts", "chart"):
            continue
        df = xl.parse(sheet)
        for col in df.select_dtypes(include=["float", "float64"]).columns:
            df[col] = df[col].round(4)
        clean_name = sheet.replace(" ", "_").replace("/", "_")
        csv_filename = f"{CSV_DIR}/{clean_name}.csv"
        df.to_csv(csv_filename, index=False, float_format="%.4f")
        exported_files.append(csv_filename)
    print(f"[csv] Exported {len(exported_files)} sheets to {CSV_DIR}/")
    return exported_files

def create_executive_html():
    import json, datetime
    exec_html_path = f"{OUT_DIR}/executive_report.html"
    summary_csv  = f"{CSV_DIR}/Summary.csv"
    formulas_csv = f"{CSV_DIR}/Formulas.csv"        # canonical name
    profile_csv  = f"{CSV_DIR}/Data_Profile.csv"    # canonical name
    run_info_path = f"{OUT_DIR}/run_info.json"

    summary_df  = pd.read_csv(summary_csv)  if os.path.exists(summary_csv)  else pd.DataFrame()
    formulas_df = pd.read_csv(formulas_csv) if os.path.exists(formulas_csv) else pd.DataFrame()
    profile_df  = pd.read_csv(profile_csv)  if os.path.exists(profile_csv)  else pd.DataFrame()

    run_info_html = ""
    if os.path.exists(run_info_path):
        try:
            ri = json.load(open(run_info_path))
            ts = ri.get('timestamp', 'N/A')
            run_info_html = f"""
            <div class="section">
              <h2 class="section-title">ℹ️ Run Information</h2>
              <table><tr><th>Field</th><th>Value</th></tr>
              <tr><td>Timestamp</td><td>{ts}</td></tr>
              <tr><td>Mode</td><td>{ri.get('mode','N/A')}</td></tr>
              <tr><td>Live Fetch</td><td>{ri.get('fetch_live','N/A')}</td></tr>
              <tr><td>Risk Aversion λ</td><td>{ri.get('risk_aversion','N/A')}</td></tr>
              <tr><td>Datasets Tested</td><td>{ri.get('total_datasets','N/A')}</td></tr>
              </table>
            </div>
            """
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Strategy &amp; Data Optimization Master Executive Report</title>
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
        <h1>📊 Quantitative Strategy &amp; Data Optimization Executive Report</h1>
        <p>Exponential Moving Average Stop-And-Reverse (EMA SAR) Multi-Asset Universe Evaluation</p>
    </div>

    {run_info_html}

    <div class="section">
        <h2 class="section-title">📐 Mathematical Formulas &amp; Methodology Documentation</h2>
        {formulas_df.to_html(classes="table", index=False) if not formulas_df.empty else "<p>No formula data available.</p>"}
    </div>

    <div class="section">
        <h2 class="section-title">🏆 Top Universe Strategy Summary (Ranked by Profit %)</h2>
        {summary_df.sort_values('profit_pct', ascending=False).head(20).to_html(classes="table", index=False) if not summary_df.empty and 'profit_pct' in summary_df.columns else (summary_df.head(20).to_html(classes="table", index=False) if not summary_df.empty else "<p>No summary data available.</p>")}
    </div>

    <div class="section">
        <h2 class="section-title">🖼️ Portfolio Universe Visual Analytics</h2>
        <div class="grid-2">
            <div>
                <h3>Universe Return vs Risk Frontier</h3>
                <img src="portfolio_rr.png" alt="Risk Return Frontier">
            </div>
            <div>
                <h3>Profit Factor Matrix (Symbol vs Timeframe)</h3>
                <img src="universe_matrix.png" alt="Universe Matrix">
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">📊 Dataset Properties &amp; Metadata</h2>
        {profile_df.to_html(classes="table", index=False) if not profile_df.empty else "<p>No dataset profile data available.</p>"}
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
