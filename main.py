import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import re
import datetime
import os
from scipy.stats import norm
from google.cloud import bigquery

# --- データ取得・解析ロジック ---

def download_jpx_data(target_date_str):
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    jst_now = utc_now + datetime.timedelta(hours=9)
    today_str = jst_now.strftime("%Y%m%d")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    df_oi, df_tp, df_settle = None, None, None
    
    # 1. 建玉情報 (Excel)
    oi_url = f"https://www.jpx.co.jp/markets/derivatives/trading-volume/tvdivq00000014nn-att/{target_date_str}open_interest.xlsx"
    res_oi = requests.get(oi_url, headers=headers)
    print(f"🔍 建玉データ取得 ({target_date_str}): ステータスコード {res_oi.status_code}")
    if res_oi.status_code == 200:
        try:
            df_oi = pd.read_excel(BytesIO(res_oi.content), sheet_name="別紙1")
            print("  - 建玉データ (Excel) 読み込み成功")
        except Exception as e:
            print(f"  - 建玉データ (Excel) 解析失敗: {e}")
        
    # 2. オプション理論価格・IV (CSV)
    tp_url = f"https://www.jpx.co.jp/automation/markets/derivatives/option-price/files/ose{target_date_str}tp.csv"
    res_tp = requests.get(tp_url, headers=headers)
    print(f"🔍 理論値・価格データ取得 ({target_date_str}): ステータスコード {res_tp.status_code}")
    if res_tp.status_code == 200:
        try: content = res_tp.content.decode('utf-8')
        except UnicodeDecodeError: content = res_tp.content.decode('shift_jis')
        try:
            df_tp = pd.read_csv(StringIO(content))
            print("  - 理論値・価格データ (CSV) 読み込み成功")
        except Exception as e:
            print(f"  - 理論値・価格データ (CSV) 解析失敗: {e}")
        
    # 3. 清算値データ (指定日付で404の場合は本日の日付でリトライ)
    for date_candidate in [target_date_str, today_str]:
        settlement_url = f"https://www.jpx.co.jp/markets/derivatives/settlement-price/tvdivq00000014l6-att/rb{date_candidate}.csv"
        res_settle = requests.get(settlement_url, headers=headers)
        print(f"🔍 清算値データ取得試行 ({date_candidate}): ステータスコード {res_settle.status_code}")
        if res_settle.status_code == 200:
            try: content = res_settle.content.decode('utf-8')
            except UnicodeDecodeError: content = res_settle.content.decode('shift_jis')
            try:
                df_settle = pd.read_csv(StringIO(content), header=None)
                print(f"  - 清算値データ (CSV: {date_candidate}) 読み込み成功")
                break
            except Exception as e:
                print(f"  - 清算値データ (CSV) 解析失敗: {e}")

    return df_oi, df_tp, df_settle

def extract_greeks_inputs(df_settle):
    if df_settle is None or df_settle.empty: return pd.DataFrame()
    try:
        df_work = df_settle.copy()
        df_work[1] = df_work[1].astype(str).str.strip()
        df_work[11] = df_work[11].astype(str).str.strip()
        df_filtered = df_work[(df_work[11] == "日経225") & (df_work[1].str.startswith("FUT_225"))].copy()
        df_filtered[3] = pd.to_numeric(df_filtered[3], errors='coerce')
        df_filtered[7] = pd.to_numeric(df_filtered[7], errors='coerce')
        df_filtered[9] = pd.to_numeric(df_filtered[9], errors='coerce')
        df_filtered[10] = pd.to_numeric(df_filtered[10], errors='coerce')
        df_filtered = df_filtered.dropna(subset=[3, 7, 9, 10])
        unique_months = sorted(df_filtered[3].unique())[:3]
        df_filtered = df_filtered[df_filtered[3].isin(unique_months)]
        df_filtered["調整残存日数"] = (df_filtered[10] - 1).clip(lower=0)
        df_inputs = pd.DataFrame({
            "限月": df_filtered[3].astype(int),
            "原資産価格_S": df_filtered[7].astype(float),
            "金利_r": df_filtered[9].astype(float),
            "残存日数_D": df_filtered["調整残存日数"].astype(int)
        })
        return df_inputs.drop_duplicates(subset=["限月"]).reset_index(drop=True)
    except Exception as e:
        print(f"  - ギリシャ指標入力値抽出エラー: {e}")
        return pd.DataFrame()

def calculate_greeks(row):
    S, K, D, v = row["原資産価格_S"], row["権利行使価格"], row["残存日数_D"], row["ボラティリティ"]
    r = row["金利_r"] / 100.0
    if D <= 0 or v <= 0: return pd.Series([0.0, 0.0, 0.0, 0.0])
    T = D / 365.0
    d1 = (np.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    pdf_d1, cdf_d1 = norm.pdf(d1), norm.cdf(d1)
    delta = cdf_d1 if row["プットコール種別"] == "call" else cdf_d1 - 1.0
    gamma = pdf_d1 / (S * v * np.sqrt(T))
    vega = (S * np.sqrt(T) * pdf_d1) / 100.0
    if row["プットコール種別"] == "call":
        theta = (- (S * v * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        theta = (- (S * v * pdf_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    return pd.Series([delta, gamma, vega, theta])

def process_data(df_oi):
    if df_oi is None or df_oi.empty: return pd.DataFrame()
    
    temp_columns = ["限月取引", "取引高", "当日建玉残高", "前日比", "前日建玉残高"]
    df_put = df_oi.iloc[:, [0, 1, 2, 3, 4]].copy()
    df_put.columns = temp_columns
    df_call = df_oi.iloc[:, [6, 7, 8, 9, 10]].copy()
    df_call.columns = temp_columns
    df_combined = pd.concat([df_put, df_call], ignore_index=True)
    
    df_combined["限月取引"] = df_combined["限月取引"].astype(str).str.strip()
    df_combined = df_combined[
        df_combined["限月取引"].str.contains("NIKKEI", na=False, case=False) & 
        ~df_combined["限月取引"].str.contains("MINI", na=False, case=False) & 
        ~df_combined["限月取引"].str.contains("合計", na=False)
    ]
    extracted = df_combined["限月取引"].str.extract(r"NIKKEI\s*225\s*([PC])(\d{4})-(\d+)", flags=re.IGNORECASE)
    df_combined["プットコール種別"] = extracted[0].str.upper().map({"P": "put", "C": "call"})
    df_combined["限月"] = extracted[1]
    df_combined["権利行使価格"] = extracted[2]
    df_combined = df_combined.dropna(subset=["プットコール種別", "限月", "権利行使価格"])
    
    df_clean = pd.DataFrame()
    df_clean["プットコール種別"] = df_combined["プットコール種別"]
    df_clean["限月"] = df_combined["限月"]
    
    num_cols = ["権利行使価格", "取引高", "当日建玉残高", "前日比", "前日建玉残高"]
    for col in num_cols:
        df_clean[col] = pd.to_numeric(
            df_combined[col].astype(str).str.replace(r'[\s,]', '', regex=True).replace('-', '0'), 
            errors='coerce'
        ).fillna(0).astype(int)
        
    return df_clean.reset_index(drop=True)

def process_tp_data(df_tp):
    if df_tp is None or df_tp.empty: return pd.DataFrame()
    headers = ["商品コード", "商品タイプ", "限月", "権利行使価格", "予備", "銘柄コード_put", "終値_put", "予備_put", "理論価格_put", "ボラティリティ_put", "銘柄コード_call", "終値_call", "予備_call", "理論価格_call", "ボラティリティ_call", "原資産終値", "基準ボラティリティ"]
    df_tp.columns = headers if len(df_tp.columns) == len(headers) else df_tp.columns
    df_filtered = df_tp[df_tp["商品コード"].astype(str).str.strip() == "NK225E"].copy()
    df_filtered["限月"] = pd.to_numeric(df_filtered["限月"], errors='coerce')
    df_filtered["権利行使価格"] = pd.to_numeric(df_filtered["権利行使価格"], errors='coerce')
    df_filtered["原資産終値"] = pd.to_numeric(df_filtered["原資産終値"], errors='coerce')
    target_months = sorted(df_filtered["限月"].dropna().unique())[:3]
    df_filtered = df_filtered[df_filtered["限月"].isin(target_months)]
    
    if df_filtered.empty: return pd.DataFrame()
    underlying = df_filtered["原資産終値"].iloc[0]
    df_filtered = df_filtered[(df_filtered["権利行使価格"] >= underlying * 0.85) & (df_filtered["権利行使価格"] <= underlying * 1.15)]
    
    df_p = df_filtered[["限月", "権利行使価格", "理論価格_put", "ボラティリティ_put", "原資産終値"]].rename(columns={"理論価格_put": "理論価格", "ボラティリティ_put": "ボラティリティ"}).copy()
    df_p["プットコール種別"] = "put"
    
    df_c = df_filtered[["限月", "権利行使価格", "理論価格_call", "ボラティリティ_call", "原資産終値"]].rename(columns={"理論価格_call": "理論価格", "ボラティリティ_call": "ボラティリティ"}).copy()
    df_c["プットコール種別"] = "call"
    
    df_res = pd.concat([df_p, df_c], ignore_index=True)
    df_res["理論価格"] = pd.to_numeric(df_res["理論価格"].astype(str).str.replace(r'[\s,]', '', regex=True).replace('-', '0'), errors='coerce').fillna(0).astype(float)
    df_res["ボラティリティ"] = pd.to_numeric(df_res["ボラティリティ"].astype(str).str.replace(r'[\s,]', '', regex=True).replace('-', '0'), errors='coerce').fillna(0).astype(float)
    df_res["限月"] = df_res["限月"].astype(int)
    df_res["権利行使価格"] = df_res["権利行使価格"].astype(int)
    return df_res

def fetch_latest_data():
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    jst_now = utc_now + datetime.timedelta(hours=9)

    # 本日の日付（JST）を取得
    today_date = jst_now.date()
    date_str = jst_now.strftime("%Y%m%d")
    
    print(f"📅 本日の日付 (JST): {date_str}")
    
    df_oi_raw, df_tp_raw, df_settle_raw = download_jpx_data(date_str)
    
    if df_oi_raw is None or df_tp_raw is None or df_settle_raw is None:
        print(f"⚠️ 本日 ({date_str}) のデータがJPX側で未公開、または一部のファイルが取得できませんでした。")
        return pd.DataFrame()
        
    df_greeks_inputs = extract_greeks_inputs(df_settle_raw)
    df_final_oi = process_data(df_oi_raw)
    df_final_tp = process_tp_data(df_tp_raw)
    
    print(f"📊 データ加工結果 - OI行数: {len(df_final_oi)}, TP行数: {len(df_final_tp)}, Inputs行数: {len(df_greeks_inputs)}")
    
    if df_final_oi.empty or df_final_tp.empty:
        print("❌ 前処理後のデータが空になりました。")
        return pd.DataFrame()
    
    df_final_oi["限月"] = pd.to_numeric("20" + df_final_oi["限月"].astype(str), errors='coerce').fillna(0).astype(int)
    df_merged = pd.merge(df_final_tp, df_final_oi, on=["プットコール種別", "限月", "権利行使価格"], how="inner")
    
    if not df_greeks_inputs.empty:
        df_merged["限月"] = df_merged["限月"].astype(int)
        df_greeks_inputs["限月"] = df_greeks_inputs["限月"].astype(int)
        df_merged = pd.merge(df_merged, df_greeks_inputs, on=["限月"], how="inner")
        
        if not df_merged.empty:
            # ギリシャ指標および GEX の計算
            df_merged[["デルタ", "ガンマ", "ベガ", "セータ"]] = df_merged.apply(calculate_greeks, axis=1)
            df_merged["GEX符号"] = df_merged["プットコール種別"].map({"call": 1.0, "put": -1.0})
            df_merged["GEX_raw"] = df_merged["ガンマ"] * df_merged["当日建玉残高"] * 1000 * df_merged["原資産価格_S"] * 0.01 * df_merged["GEX符号"]
            df_merged["GEX_oku"] = df_merged["GEX_raw"] / 100000000.0
            
            # 日付カラムを追加
            df_merged["data_date"] = today_date
            
            # BigQuery用カラム名変換
            rename_map = {
                "限月": "expiry_month",
                "権利行使価格": "strike_price",
                "プットコール種別": "put_call_type",
                "理論価格": "theoretical_price",
                "ボラティリティ": "volatility",
                "原資産終値": "underlying_close",
                "取引高": "volume",
                "当日建玉残高": "open_interest",
                "前日比": "oi_change",
                "前日建玉残高": "prev_open_interest",
                "原資産価格_S": "futures_settlement",
                "金利_r": "interest_rate",
                "残存日数_D": "days_to_expiry",
                "デルタ": "delta",
                "ガンマ": "gamma",
                "ベガ": "vega",
                "セータ": "theta",
                "GEX符号": "gex_sign",
                "GEX_raw": "gex_raw",
                "GEX_oku": "gex_oku"
            }
            
            df_bq = df_merged.rename(columns=rename_map)
            
            # --- ストライクごとの横持ち（ピボット集計）処理 ---
            df_bq['call_oi'] = np.where(df_bq['put_call_type'] == 'call', df_bq['open_interest'], 0)
            df_bq['put_oi'] = np.where(df_bq['put_call_type'] == 'put', df_bq['open_interest'], 0)
            df_bq['call_gex_oku'] = np.where(df_bq['put_call_type'] == 'call', df_bq['gex_oku'], 0)
            df_bq['put_gex_oku'] = np.where(df_bq['put_call_type'] == 'put', df_bq['gex_oku'], 0)

            group_cols = ['data_date', 'expiry_month', 'strike_price', 'underlying_close', 'futures_settlement', 'interest_rate', 'days_to_expiry']
            df_grouped = df_bq.groupby(group_cols, as_index=False).agg({
                'call_oi': 'sum',
                'put_oi': 'sum',
                'volume': 'sum',
                'call_gex_oku': 'sum',
                'put_gex_oku': 'sum',
                'gex_oku': 'sum'  # ネットGEX (Call GEX + Put GEX)
            })

            df_grouped['net_open_interest'] = df_grouped['call_oi'] - df_grouped['put_oi']
            df_grouped = df_grouped.rename(columns={'gex_oku': 'net_gex_oku'})

            return df_grouped
            
    print("❌ ギリシャ指標計算・マージ後のデータが空になりました。")
    return pd.DataFrame()


# --- BigQuery 格納処理 ---

def save_to_bigquery(df):
    if df.empty:
        print("⚠️ 格納するデータが存在しません（処理をスキップします）。")
        return

    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
    DATASET_ID = "jpx_options"
    TABLE_ID = os.getenv("TABLE_NAME", "gex_daily_pivot")
    TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    print(f"🚀 BigQuery ({TABLE_REF}) へデータを書き込み中... 基準日: {df['data_date'].iloc[0]}")

    client = bigquery.Client(project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="data_date"
        )
    )

    try:
        job = client.load_table_from_dataframe(
            df, TABLE_REF, job_config=job_config
        )
        job.result()  # 完了待機
        print("✅ BigQueryへのデータ格納が正常に完了しました！")
    except Exception as e:
        print(f"❌ BigQueryへの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    df_to_save = fetch_latest_data()
    save_to_bigquery(df_to_save)