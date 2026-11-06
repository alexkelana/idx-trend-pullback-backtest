import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from backtest_engine import Config, backtest, pf_r, boot_ci

st.set_page_config(page_title="IDX Trend-Pullback Backtest", layout="wide")
st.title("IDX Trend-Pullback Backtest")
st.caption("Supertrend pullback + fractal structure + konfirmasi N dari 6. Long only, lot 100.")

with st.sidebar:
    st.header("Sumber Data")
    source = st.radio("Pilih sumber", ["Yahoo Finance", "Upload CSV (Google Finance)"])
    tickers_input = st.text_input("Ticker (pisah koma)", "BBNI.JK,BBRI.JK,BMRI.JK,BBCA.JK")
    start_date = st.date_input("Mulai", date(2015, 1, 1))
    end_date = st.date_input("Akhir", date.today())
    min_conf = st.selectbox("Minimal konfirmasi", [0, 2, 3, 4], index=2)
    fee = st.number_input("Fee per sisi", 0.0, 0.01, 0.002, format="%.4f")
    slip = st.number_input("Slippage per sisi", 0.0, 0.01, 0.001, format="%.4f")
    run = st.button("Jalankan Backtest", type="primary")

st.info("""
**Google Finance**: Tidak ada API publik. Cara terbaik: di Google Sheets ketik  
`=GOOGLEFINANCE("IDX:BBNI","all",DATE(2015,1,1),TODAY())`  
lalu File → Download → CSV, kemudian upload di sini.
""")

@st.cache_data(show_spinner=False)
def load_yahoo(tickers, start, end):
    frames = {}
    for t in [x.strip() for x in tickers.split(",") if x.strip()]:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            frames[t] = df[["open", "high", "low", "close", "volume"]].dropna()
    return frames

def load_csv(uploaded):
    frames = {}
    for f in uploaded:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = next((c for c in ("date", "time", "datetime", "tanggal") if c in df.columns), None)
        if tcol is None:
            st.error(f"{f.name}: tidak ada kolom tanggal")
            continue
        df[tcol] = pd.to_datetime(df[tcol])
        df = df.set_index(tcol)
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                st.error(f"{f.name}: kurang kolom {col}")
                continue
        if "volume" not in df.columns:
            df["volume"] = 0.0
        name = f.name.replace(".csv", "").upper()
        frames[name] = df[["open", "high", "low", "close", "volume"]].dropna()
    return frames

if run:
    if source == "Yahoo Finance":
        data = load_yahoo(tickers_input, start_date, end_date)
    else:
        uploaded = st.file_uploader("Upload CSV dari Google Sheets", accept_multiple_files=True, type="csv")
        if not uploaded:
            st.warning("Upload dulu file CSV-nya.")
            st.stop()
        data = load_csv(uploaded)

    if not data:
        st.error("Tidak ada data yang berhasil dimuat.")
        st.stop()

    cfg = Config(min_conf=min_conf, fee=fee, slip=slip)
    all_trades = []
    for name, df in data.items():
        if len(df) < 400:
            st.warning(f"{name}: data terlalu pendek ({len(df)} bar)")
            continue
        tr = backtest(df, cfg, name)
        if len(tr):
            all_trades.append(tr)
            st.success(f"{name}: {len(tr)} trade")

    if not all_trades:
        st.error("Tidak ada trade yang dihasilkan.")
        st.stop()

    trades = pd.concat(all_trades, ignore_index=True)
    r = trades["r"].to_numpy()
    lo, hi = boot_ci(r)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Trade", len(trades))
    col2.metric("Win Rate", f"{(trades.pnl_pct > 0).mean()*100:.1f}%")
    col3.metric("Expectancy (R)", f"{r.mean():.3f}")
    col4.metric("PF (R)", f"{pf_r(r):.2f}")

    st.subheader("95% CI Expectancy")
    st.write(f"[{lo:.2f}, {hi:.2f}]  ← batas bawah harus > 0 agar dianggap positif")

    eq = (1 + trades.sort_values("exit_time")["pnl_pct"]).cumprod()
    st.line_chart(eq)

    st.subheader("Trade Terakhir")
    st.dataframe(trades.tail(20)[["ticker", "entry_time", "exit_time", "reason", "r", "pnl_pct", "conf"]], use_container_width=True)

    st.download_button("Download semua trade (CSV)", trades.to_csv(index=False), "trades.csv")

    st.warning("""
    **Catatan penting**  
    - Bootstrap CI mengasumsikan trade independen → biasanya terlalu optimis.  
    - Minimal ~150 trade baru layak dilihat serius.  
    - Data Yahoo bisa punya artefak corporate action. CSV dari Google Sheets lebih bersih.  
    - Ini hasil historis, bukan jaminan masa depan.
    """)
else:
    st.write("Atur parameter di sidebar lalu klik **Jalankan Backtest**.")
