# 📈 Web Forecasting - Aktualisasi Latsar CPNS 2025

A web-based forecasting application developed as part of the CPNS 2025 Basic Training (Latsar) actualization project. This tool allows users to upload time series data, configure forecasting parameters, evaluate model performance, and export forecasting results in a user-friendly interface.

---

## 📋 Table of Contents

- [📈 Features](#-features)
- [📦 Usage](#-usage)
- [🧠 Forecasting Methods Supported](#-forecasting-methods-supported)
- [🎥 Demo](#-demo)
- [🗂️ Project Structure](#-project-structure)
- [👤 Author](#-author)
- [🙏 Special Thanks](#-special-thanks)
- [📄 License](#-license)
- [📚 Notes](#-notes)

---

## 📈 Features

- ✅ Upload Excel (.xlsx) or CSV (.csv) time series data  
- ✅ Flexible forecasting settings:
  - Choose variable/column to forecast
  - Select forecasting method (e.g., Simple Exponential Smoothing, Holt, ARIMA)
  - Set number of forecasted periods  
- ✅ Model Evaluation:
  - Visualize actual vs forecast data
  - RMSE score for accuracy check  
- ✅ Forecast Result:
  - Display forecast result in table format
  - Plot forecast output
  - Export forecast result to downloadable file  

---

## 📦 Usage

1. Upload your `.xlsx` or `.csv` file containing time series data.
2. Choose which column to forecast and select the forecasting method.
3. Input the number of periods to forecast.
4. Click **Run Forecast**.
5. View the plot, RMSE value, and resulting forecast table.
6. Download the result as Excel.

---

## 🧠 Forecasting Methods Supported

- Simple Exponential Smoothing  
- Holt’s Linear Trend Method  
- ARIMA (AutoRegressive Integrated Moving Average)  

---

## 🎥 Demo

🔗 [Watch the Demo](https://youtu.be/ClxcSgtwIiA)

---

## 🗂️ Project Structure

```
web-forecasting/
├── base.py
├── requirements.txt
├── README.md
```

---

## 👤 Author

**Mohamad Hafidz Al Habib**  
📍 CPNS Kabupaten Tulungagung Angkatan 2025  
📧 hafidzalhabib@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mohamad-hafidz-al-habib-99609b237)

---

## 🙏 Special Thanks

- Bapak **Widyaiswara** atas bimbingan selama pelatihan dasar  
- Ibu **Mentor** dan rekan sejawat yang telah memberikan masukan dan semangat  
- Komunitas **Streamlit** dan **Statsmodels** atas dokumentasi dan pustaka yang sangat membantu  

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.

---

## 📚 Notes

Project ini dikembangkan sebagai bagian dari tugas aktualisasi Latsar CPNS 2025. Aplikasi ini diharapkan dapat digunakan untuk kebutuhan analisis dan perencanaan berbasis data di instansi pemerintahan maupun sektor lainnya.
