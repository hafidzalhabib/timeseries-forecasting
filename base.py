import streamlit as st
# from streamlit.components.v1 import html
import pandas as pd
import openpyxl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
# setting config
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://wa.me/6285536956301',
        'Report a bug': "https://wa.me/6285536956301",
        'About': "## Forecasting App\n\nWebsite ini dibuat untuk aktualisasi Pelatihan Dasar CPNS Tahun 2025"
    }
)
# setting CSS
st.markdown("""
    <style>
        .custom-header {
            padding: 0;
            display: flex;
            align-items: center;
            padding: 10px;
            background-color: var(--background-color);
            color: var(--text-color);
            border-bottom: 2px solid var(--secondary-background-color);
        }
        .custom-header img {
            height: 65px;
            margin-right: 15px;
        }
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: black;
            border-top: 1px solid grey;
            z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)
# Header
st.markdown("""
    <div class="custom-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/Lambang-tulungagung.png">
        <div>
            <div style='font-weight: 900; font-size: 24px; margin-bottom: 0px;'>BPKAD Kabupaten Tulungagung</div>
            <div style='font-weight: 600; font-size: 16px; margin-top: 0px;'>Bidang Anggaran</div>
        </div>
    </div>
    <hr style="border: none; height: 2px; background-color: #ccc; margin-top: 10px;">
""", unsafe_allow_html=True)
# Upload file
st.subheader("Upload Data")
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx", "csv"])
st.caption("Panduan [klik disini](%s)" % "https://youtu.be/ClxcSgtwIiA")
if uploaded_file is not None:
    if uploaded_file.name != st.session_state["uploaded_filename"]:
        st.toast(f"✅ File berhasil diunggah: {uploaded_file.name}")
        st.session_state["file_uploaded"] = True
        st.session_state["uploaded_filename"] = uploaded_file.name
else:
    st.session_state["file_uploaded"] = False
    st.session_state["uploaded_filename"] = None
# membaca file
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, parse_dates=True)
        # formatting data
        def formatting(df):
            # Deteksi kolom pertama
            kolom_pertama = df.columns[0]
            # Kolom-kolom yang ingin diformat
            kolom_format = [
                col for col in df.select_dtypes(include='number').columns
                if col != kolom_pertama
            ]
            # Terapkan formatting hanya ke kolom numerik selain kolom pertama
            styled_df = df.style.format({col: '{:,.0f}' for col in kolom_format})
            return styled_df
        # Tampilkan data
        st.dataframe(formatting(df))
        # Pilih Variabel
        col1, col2 = st.columns(2)
        with col1:
            tahun = st.selectbox("Pilih variabel tahun:", df.columns)
        with col2:
            var = st.selectbox("Pilih variabel forecasting:", df.columns)
        # Pilih metode dan input forecast
        col21, col22 = st.columns(2)
        with col21:
            select_model = st.selectbox("Pilih metode yang digunakan:", ("Simple Exponential Smoothing", "Holt's Exponential Smoothing", "ARIMA"))
        with col22:
            forecast = st.number_input(label="Masukkan jumlah forecast", min_value=0, max_value=100)
        # Buat fix dataset
        dataset = df[[tahun, var]]
        # dataset[tahun] = pd.to_datetime(dataset[tahun], errors="coerce")
        dataset[var] = pd.to_numeric(dataset[var], errors='coerce')
        dataset = dataset.set_index(tahun)
        # split train dan test data
        df_train = dataset.iloc[:round(len(dataset) * 0.8)]
        df_test = dataset.iloc[round(len(dataset) * 0.8):]
        # gridsearch simple exponential smoothing
        def grid_ses(train, test):
            best_score = float("inf")
            best_alpha = None
            results = []
            alphas = np.arange(0.01, 1.01, 0.01)  # rentang smoothing_level
            for alpha in alphas:
                model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
                forecast = model.forecast(len(test))
                mse = mean_squared_error(test, forecast)
                results.append((alpha, mse))
                if mse < best_score:
                    best_score = mse
                    best_alpha = alpha
            return best_score, best_alpha
        # gridsearch holt's exponential smoothing
        def grid_holts(train, test):
            best_score = float("inf")
            best_alpha, best_beta = None, None
            results = []
            alpha_range = np.arange(0.01, 1.01, 0.1)
            beta_range = np.arange(0.01, 1.01, 0.1)
            for alpha in alpha_range:
                for beta in beta_range:
                    try:
                        model = Holt(train).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)
                        forecast = model.forecast(len(test))
                        mse = mean_squared_error(test, forecast)
                        results.append((alpha, beta, mse))
                        if mse < best_score:
                            best_score = mse
                            best_alpha = alpha
                            best_beta = beta
                    except:
                        continue  # skip if model fails to fit with given params
            return best_score, best_alpha, best_beta
        # gridsearch arima
        def grid_arima(train, test):
            p_values = range(0, 4)
            d_values = range(0, 2)
            q_values = range(0, 4)
            best_score = float("inf")
            best_cfg = None
            results = []
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(train, order=(p, d, q)).fit()
                            forecast = model.forecast(steps=len(test))
                            mse = mean_squared_error(test, forecast)
                            results.append(((p, d, q), mse))
                            if mse < best_score:
                                best_score = mse
                                best_cfg = (p, d, q)
                        except:
                            continue
            return best_score, best_cfg
        # ambil parameter hasil gridsearch
        def param(train, test):
            alpha = None
            beta = None
            pdq = None
            if select_model == "Simple Exponential Smoothing":
                mse, alpha = grid_ses(train, test)
            if select_model == "Holt's Exponential Smoothing":
                mse, alpha, beta = grid_holts(train, test)
            if select_model == "ARIMA":
                mse, pdq = grid_arima(train, test)
            return alpha, beta, pdq, mse
        # def forcasting
        def forecasting(model, fix_data, fix_forecast, alpha, beta, pdq):
            if model == "Simple Exponential Smoothing":
                fix_model = SimpleExpSmoothing(fix_data).fit(smoothing_level=alpha, optimized=False)
            if model == "Holt's Exponential Smoothing":
                fix_model = Holt(fix_data).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)
            if model == "ARIMA":
                fix_model = ARIMA(fix_data, order=pdq).fit()
            forecasting = fix_model.forecast(fix_forecast)
            return forecasting
        # tombol proses data
        pilih = st.button('**PILIH**')
        if pilih:
            with st.spinner("⏳ Sedang memproses data. Sabar ya😊😅..."):
                # Buat konversi ke nilai miliar
                def miliar_format(x, pos):
                    return f'{x / 1e9:,.1f} M'
                # Ambil parameter hasil grid search
                alpha, beta, pdq, mse = param(df_train, df_test)
                # forecast data train
                forecast_train = forecasting(select_model, df_train, len(df_test), alpha, beta, pdq)
                # Visualisasi data train
                fig = plt.figure(figsize=(10, 5))
                fig.suptitle(f'{var} Tahun {int(dataset.index.min())} - {int(dataset.index.max())}', fontsize=12)
                past, = plt.plot(df_train.index, df_train, "b.-", label= f'History {var}')
                future, = plt.plot(df_test.index, df_test, "r.-", label= f'Aktual {var}')
                pred_future, = plt.plot(df_test.index, forecast_train, "g.--", label= f'Forecasting {var}')
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(miliar_format))
                plt.xlabel(f'{tahun}', fontsize=10)
                plt.ylabel(f'{var} (Miliar Rupiah)', fontsize=10)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(handles=[past, future, pred_future], loc="upper left")
                plt.text(0.025, 0.75, f"RMSE: {np.sqrt(mse) / 1e9:,.1f} M",
                         transform=plt.gca().transAxes,
                         fontsize=10,
                         color='black',
                         verticalalignment='top')
                # forecast data
                forecast_rill = forecasting(select_model, dataset, forecast, alpha, beta, pdq)
                # buat index untuk hasil forecast
                last = int(dataset.index.max())
                pred_index = [int(i) for i in range(last + 1, last + forecast + 1)]
                # buat tabel hasil forecast
                tabel_forecast = pd.DataFrame({
                    tahun : pred_index,
                    var : forecast_rill
                })
                tabel_forecast = tabel_forecast[[tahun, var]].set_index(tahun)
                # visualisasi hasil forecast
                fig2 = plt.figure(figsize=(10, 5))
                if forecast ==1:
                    fig2.suptitle(
                        f'Forecasting {var} Tahun {min(pred_index)}',
                        fontsize=12)
                else:
                    fig2.suptitle(f'Forecasting {var} Tahun {min(pred_index)} - {max(pred_index)}', fontsize=12)
                dataset.index = dataset.index.astype(int)
                past2, = plt.plot(dataset.index, dataset, "b.-", label= f'History {var}')
                future2, = plt.plot(pred_index, forecast_rill, "g.--", label= f'Forecasting {var}')
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(miliar_format))
                plt.xlabel(f'{tahun}', fontsize=10)
                plt.ylabel(f'{var} (Miliar Rupiah)', fontsize=10)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(handles=[past2, future2], loc="upper left")
            # Plot uji kebaikan model
            st.subheader("Uji Kebaikan Metode")
            st.pyplot(fig)
            # Plot hasil forecasting
            st.subheader("Hasil Forecasting")
            st.dataframe(tabel_forecast.style.format({var: '{:,.0f}'}))
            st.pyplot(fig2)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
# footer
st.markdown("""
    <div class="custom-footer">
        © 2025 BPKAD Kabupaten Tulungagung - Dibuat untuk Aktualisasi Latsar CPNS Tahun 2025
    </div>
""", unsafe_allow_html=True)
