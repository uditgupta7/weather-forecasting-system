# 🌤️ WeatherScope Pro

> A professional, ML-powered Weather Forecasting System built with Python & Streamlit.  

---

## 📸 Dashboard Preview

![WeatherScope Pro Dashboard](screenshots/dashboard.png)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **City Search** | Real-time weather lookup by city name |
| 🌡️ **Current Conditions** | Temperature, humidity, wind, visibility, feels-like |
| 📅 **5-Day Forecast** | Daily weather cards with emoji icons |
| 📈 **Trend Charts** | Temperature, precipitation, humidity, wind charts (Matplotlib) |
| 🤖 **ML Prediction** | 7-day temperature forecast via Random Forest Regressor |
| 🌬️ **Air Quality Index** | AQI level + CO, NO₂, O₃, PM2.5, PM10, SO₂ breakdown |
| ⚠️ **Weather Alerts** | Automatic alerts for extreme heat, wind, visibility, storms |
| 🌅 **Sunrise/Sunset** | Calculated in local timezone |
| 🕓 **Search History** | Saved locally (JSON), clickable sidebar |
| 🎨 **Dark / Light Mode** | Toggle in sidebar |
| 🌡️ **°C / °F Toggle** | Unit conversion built-in |

---

## 🛠 Tech Stack

```
Language  : Python 3.10+
Frontend  : Streamlit
ML Model  : Scikit-learn (Random Forest Regressor)
Charts    : Matplotlib
Data      : Pandas, NumPy
API       : OpenWeatherMap (free tier)
Styling   : Custom CSS (dark + light themes)
```

---

## 📂 Project Structure

```
weather-forecasting-system/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── assets/
│   ├── dark_theme.css      # Dark mode styles
│   └── light_theme.css     # Light mode styles
│
├── data/
│   └── history.json        # Auto-generated search history
│
├── models/                 # (Future: save trained ML models)
│
├── utils/
│   ├── __init__.py
│   ├── weather_api.py      # OpenWeatherMap API wrapper
│   ├── ml_predictor.py     # Random Forest temperature predictor
│   ├── history_manager.py  # City search history (read/write JSON)
│   └── helpers.py          # Emoji map, AQI, wind direction, etc.
│
└── screenshots/            # Add screenshots here for README
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/weather-forecasting-system.git
cd weather-forecasting-system
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free API key

1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Sign up for a free account
3. Go to **My API Keys** and copy your key
4. The free tier includes: Current Weather, 5-Day Forecast, AQI — all used here

### 5. Set your API key

**Option A – Environment variable (recommended)**

```bash
# Windows
set OPENWEATHER_API_KEY=your_api_key_here

# macOS / Linux
export OPENWEATHER_API_KEY=your_api_key_here
```

**Option B – Edit app.py directly**

Open `app.py` and replace:
```python
api_key = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
```
with:
```python
api_key = "your_actual_api_key_here"
```

### 6. Run the app

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501** 🎉

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push your project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select your repo → set `app.py` as the main file
4. Add your API key under **Settings → Secrets**:

```toml
# .streamlit/secrets.toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

Then in `app.py`, change:
```python
api_key = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
```
to:
```python
api_key = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", ""))
```

---

## 🤖 Machine Learning Details

**Model:** `sklearn.ensemble.RandomForestRegressor`  
**Training Data:** Current 5-day API forecast (40 data points)  
**Features:**
- Hour of day
- Day of week
- Month
- Humidity (%)
- Atmospheric pressure (hPa)
- Wind speed (m/s)

**Target:** Temperature (°C)  
**Prediction:** Daily average temperatures for the next 7 days

---

## 🔮 Future Improvements

- [ ] Historical weather data visualization
- [ ] Weather map with Folium
- [ ] Push notifications for alerts
- [ ] LSTM-based deep learning model
- [ ] Multi-city comparison dashboard
- [ ] Offline mode with cached data
- [ ] Export data as CSV/PDF report
- [ ] PWA support for mobile

---

## 📜 License

MIT License — free to use and modify.

---

## 🙏 Acknowledgements

- [OpenWeatherMap](https://openweathermap.org/) — free weather API
- [Streamlit](https://streamlit.io/) — rapid Python web apps
- [Scikit-learn](https://scikit-learn.org/) — ML toolkit

---

*Built with ❤️ using Python, Streamlit, and Machine Learning*
