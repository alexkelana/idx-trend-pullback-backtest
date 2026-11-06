# IDX Trend-Pullback Backtest (Streamlit)

Frontend Streamlit untuk backtest strategi Trend-Pullback + konfirmasi (N dari 6) di saham IDX.

## Cara pakai

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sumber data

- **Google Finance (disarankan untuk akurasi)**: Buat Google Sheet dengan formula  
  `=GOOGLEFINANCE("IDX:BBNI","all",DATE(2010,1,1),TODAY())`  
  lalu export ke CSV dan upload di aplikasi.
- **Yahoo Finance**: Fallback otomatis (ticker `.JK`). Data sudah auto-adjust.

## Catatan

Hasil backtest hanya untuk riset. Bukan nasihat keuangan.  
Lihat peringatan di dalam aplikasi tentang sample size, bootstrap, dan kualitas data.
