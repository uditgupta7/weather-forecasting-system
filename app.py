"""
Weather Forecasting System
==========================
A professional weather dashboard with ML-based predictions,
real-time API data, charts, and AQI support.

Author: Weather Forecasting System
Tech Stack: Python, Streamlit, Pandas, Matplotlib, Scikit-learn
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# ── internal modules ──────────────────────────────────────────
from utils.weather_api   import WeatherAPI
from utils.ml_predictor  import WeatherPredictor
from utils.history_manager import HistoryManager
from utils.helpers       import (
    get_weather_emoji,
    celsius_to_fahrenheit,
    get_aqi_label,
    get_wind_direction,
    format_sunrise_sunset,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WeatherScope Pro",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# LOAD CUSTOM CSS
# ══════════════════════════════════════════════════════════════
def load_css(theme: str = "dark") -> None:
    css_file = Path(__file__).parent / "assets" / f"{theme}_theme.css"
    if css_file.exists():
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════
def init_session_state() -> None:
    defaults = {
        "theme":        "dark",
        "unit":         "Celsius",
        "city_history": [],
        "current_data": None,
        "forecast_data": None,
        "aqi_data":     None,
        "last_city":    "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar(api: WeatherAPI, history: HistoryManager) -> str | None:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # theme toggle
        theme_choice = st.radio(
            "🎨 Theme", ["Dark", "Light"],
            index=0 if st.session_state.theme == "dark" else 1,
            horizontal=True,
        )
        st.session_state.theme = theme_choice.lower()

        # unit toggle
        st.session_state.unit = st.radio(
            "🌡️ Temperature Unit", ["Celsius", "Fahrenheit"],
            horizontal=True,
        )

        st.markdown("---")
        st.markdown("## 🔍 Search City")
        city_input = st.text_input(
            "Enter city name",
            placeholder="e.g. Mumbai, London, New York",
        )
        search_btn = st.button("🔎 Get Weather", use_container_width=True)

        # recent searches
        saved = history.load()
        if saved:
            st.markdown("---")
            st.markdown("### 🕓 Recent Searches")
            for city in saved[-6:][::-1]:
                if st.button(f"📍 {city}", key=f"hist_{city}", use_container_width=True):
                    return city

        st.markdown("---")
        st.markdown(
            "<small>Built with ❤️ using Python & Streamlit<br>"
            "Data: OpenWeatherMap API</small>",
            unsafe_allow_html=True,
        )

    if search_btn and city_input.strip():
        return city_input.strip()
    return None

# ══════════════════════════════════════════════════════════════
# CURRENT WEATHER CARD
# ══════════════════════════════════════════════════════════════
def render_current_weather(data: dict, unit: str) -> None:
    temp     = data["main"]["temp"]
    feels    = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind     = data["wind"]["speed"]
    vis      = data.get("visibility", 0) / 1000
    desc     = data["weather"][0]["description"].title()
    country  = data["sys"]["country"]
    city     = data["name"]
    emoji    = get_weather_emoji(data["weather"][0]["main"])
    w_dir    = get_wind_direction(data["wind"].get("deg", 0))

    if unit == "Fahrenheit":
        temp  = celsius_to_fahrenheit(temp)
        feels = celsius_to_fahrenheit(feels)
        sym   = "°F"
    else:
        sym = "°C"

    sunrise = format_sunrise_sunset(data["sys"]["sunrise"], data["timezone"])
    sunset  = format_sunrise_sunset(data["sys"]["sunset"],  data["timezone"])

    st.markdown(
        f"""
        <div class="weather-card main-card">
            <div class="city-header">
                <span class="city-name">📍 {city}, {country}</span>
                <span class="update-time">Updated: {datetime.now().strftime('%H:%M %d %b %Y')}</span>
            </div>
            <div class="temp-section">
                <span class="weather-emoji">{emoji}</span>
                <span class="temperature">{temp:.1f}{sym}</span>
                <span class="condition">{desc}</span>
            </div>
            <div class="feels-like">Feels like {feels:.1f}{sym}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "💧 Humidity",    f"{humidity}%"),
        (c2, "💨 Wind",        f"{wind} m/s {w_dir}"),
        (c3, "👁️ Visibility",  f"{vis:.1f} km"),
        (c4, "🌡️ Feels Like", f"{feels:.1f}{sym}"),
        (c5, "🌅 Sunrise",     sunrise),
        (c6, "🌇 Sunset",      sunset),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════
# 5-DAY FORECAST
# ══════════════════════════════════════════════════════════════
def render_forecast(forecast_data: dict, unit: str) -> None:
    st.markdown("### 📅 5-Day Forecast")
    items = forecast_data["list"]

    # one entry per day (noon)
    daily: dict[str, list] = {}
    for item in items:
        date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
        daily.setdefault(date, []).append(item)

    cols = st.columns(min(5, len(daily)))
    for idx, (date, entries) in enumerate(list(daily.items())[:5]):
        avg_temp = np.mean([e["main"]["temp"] for e in entries])
        avg_hum  = np.mean([e["main"]["humidity"] for e in entries])
        desc     = entries[len(entries)//2]["weather"][0]["description"].title()
        emoji    = get_weather_emoji(entries[len(entries)//2]["weather"][0]["main"])
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%a, %d %b")

        if unit == "Fahrenheit":
            avg_temp = celsius_to_fahrenheit(avg_temp)
            sym = "°F"
        else:
            sym = "°C"

        with cols[idx]:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_name}</div>
                    <div class="forecast-emoji">{emoji}</div>
                    <div class="forecast-temp">{avg_temp:.1f}{sym}</div>
                    <div class="forecast-desc">{desc}</div>
                    <div class="forecast-humidity">💧 {avg_hum:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════
# TEMPERATURE TREND CHART
# ══════════════════════════════════════════════════════════════
def render_temperature_chart(forecast_data: dict, unit: str, theme: str) -> None:
    st.markdown("### 📈 Temperature Trend (Next 40 Hours)")

    items  = forecast_data["list"][:14]          # ~40 h
    times  = [datetime.fromtimestamp(i["dt"]) for i in items]
    temps  = [i["main"]["temp"] for i in items]
    feels  = [i["main"]["feels_like"] for i in items]
    precip = [i.get("pop", 0) * 100 for i in items]

    if unit == "Fahrenheit":
        temps = [celsius_to_fahrenheit(t) for t in temps]
        feels = [celsius_to_fahrenheit(f) for f in feels]
        sym   = "°F"
    else:
        sym = "°C"

    bg    = "#0e1117" if theme == "dark" else "#ffffff"
    fc    = "#fafafa" if theme == "dark" else "#111111"
    grid  = "#333333" if theme == "dark" else "#dddddd"

    fig, ax1 = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(bg)
    ax1.set_facecolor(bg)

    ax1.plot(times, temps, color="#00d4ff", linewidth=2.5,
             marker="o", markersize=5, label=f"Temp ({sym})")
    ax1.plot(times, feels, color="#ff6b6b", linewidth=1.5,
             linestyle="--", label=f"Feels Like ({sym})")
    ax1.fill_between(times, temps, alpha=0.1, color="#00d4ff")

    ax1.set_ylabel(f"Temperature ({sym})", color=fc, fontsize=11)
    ax1.tick_params(colors=fc)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, color=fc, fontsize=8)
    ax1.grid(True, color=grid, alpha=0.4, linestyle="--")
    ax1.spines[:].set_color(grid)

    ax2 = ax1.twinx()
    ax2.bar(times, precip, width=0.07, alpha=0.4, color="#7b2ff7", label="Rain %")
    ax2.set_ylabel("Precipitation %", color=fc, fontsize=11)
    ax2.tick_params(colors=fc)
    ax2.spines[:].set_color(grid)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=bg, labelcolor=fc, fontsize=9, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
# AQI SECTION
# ══════════════════════════════════════════════════════════════
def render_aqi(aqi_data: dict) -> None:
    if not aqi_data:
        return
    st.markdown("### 🌬️ Air Quality Index")

    aqi   = aqi_data["list"][0]["main"]["aqi"]
    comps = aqi_data["list"][0]["components"]
    label, color = get_aqi_label(aqi)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f"""
            <div class="aqi-badge" style="border-color:{color}">
                <div class="aqi-number" style="color:{color}">{aqi}</div>
                <div class="aqi-label" style="color:{color}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        pollutants = {
            "CO":    comps.get("co",    0),
            "NO₂":   comps.get("no2",   0),
            "O₃":    comps.get("o3",    0),
            "PM2.5": comps.get("pm2_5", 0),
            "PM10":  comps.get("pm10",  0),
            "SO₂":   comps.get("so2",   0),
        }
        cols = st.columns(len(pollutants))
        for col, (name, val) in zip(cols, pollutants.items()):
            with col:
                st.markdown(
                    f'<div class="pollutant-card"><b>{name}</b><br>{val:.1f} µg/m³</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════
# ML PREDICTION SECTION
# ══════════════════════════════════════════════════════════════
def render_ml_prediction(forecast_data: dict, unit: str, theme: str) -> None:
    st.markdown("### 🤖 ML Temperature Prediction (Next 7 Days)")

    predictor = WeatherPredictor()
    items = forecast_data["list"]

    # build training dataframe from forecast
    records = []
    for item in items:
        dt = datetime.fromtimestamp(item["dt"])
        records.append({
            "hour":     dt.hour,
            "day":      dt.weekday(),
            "month":    dt.month,
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind":     item["wind"]["speed"],
            "temp":     item["main"]["temp"],
        })
    df = pd.DataFrame(records)

    predictions, dates = predictor.predict_next_days(df, days=7)

    if unit == "Fahrenheit":
        predictions = [celsius_to_fahrenheit(t) for t in predictions]
        sym = "°F"
    else:
        sym = "°C"

    bg   = "#0e1117" if theme == "dark" else "#ffffff"
    fc   = "#fafafa" if theme == "dark" else "#111111"
    grid = "#333333" if theme == "dark" else "#dddddd"

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(dates, predictions, color="#f5a623", linewidth=2.5,
            marker="D", markersize=7, label="ML Predicted Temp")
    ax.fill_between(dates, predictions, alpha=0.15, color="#f5a623")

    for x, y in zip(dates, predictions):
        ax.annotate(f"{y:.1f}{sym}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", color=fc, fontsize=8)

    ax.set_ylabel(f"Temperature ({sym})", color=fc, fontsize=11)
    ax.tick_params(colors=fc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a, %d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, color=fc, fontsize=9)
    ax.grid(True, color=grid, alpha=0.4, linestyle="--")
    ax.spines[:].set_color(grid)
    ax.legend(facecolor=bg, labelcolor=fc, fontsize=9)
    ax.set_title("7-Day ML Forecast", color=fc, fontsize=13, pad=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "<small>🔬 Model: Random Forest Regressor trained on forecast data. "
        "Features: hour, day, month, humidity, pressure, wind speed.</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# WEATHER ALERTS SECTION
# ══════════════════════════════════════════════════════════════
def render_alerts(current_data: dict) -> None:
    alerts = []
    temp     = current_data["main"]["temp"]
    humidity = current_data["main"]["humidity"]
    wind     = current_data["wind"]["speed"]
    vis      = current_data.get("visibility", 10000) / 1000
    cond     = current_data["weather"][0]["main"].lower()

    if temp >= 40:
        alerts.append(("🔴 Extreme Heat Alert", f"Temperature is {temp:.1f}°C. Stay hydrated.", "red"))
    elif temp >= 35:
        alerts.append(("🟠 Heat Warning", f"High temperature {temp:.1f}°C. Avoid midday sun.", "orange"))
    elif temp <= 0:
        alerts.append(("🔵 Freezing Alert", f"Temperature below 0°C. Risk of frost/ice.", "blue"))

    if wind >= 20:
        alerts.append(("💨 Strong Wind Alert", f"Wind speed {wind:.1f} m/s. Secure loose items.", "orange"))
    if humidity >= 90:
        alerts.append(("💧 High Humidity Warning", f"Humidity at {humidity}%. Feels muggy.", "yellow"))
    if vis < 1:
        alerts.append(("🌫️ Low Visibility", f"Visibility {vis:.1f} km. Drive carefully.", "yellow"))
    if "storm" in cond or "thunder" in cond:
        alerts.append(("⛈️ Thunderstorm Warning", "Thunderstorm conditions. Stay indoors.", "red"))

    if alerts:
        st.markdown("### ⚠️ Weather Alerts")
        for title, msg, _ in alerts:
            st.warning(f"**{title}** — {msg}")
    else:
        st.success("✅ No active weather alerts for this location.")

# ══════════════════════════════════════════════════════════════
# HUMIDITY & WIND CHARTS
# ══════════════════════════════════════════════════════════════
def render_humidity_wind_chart(forecast_data: dict, theme: str) -> None:
    st.markdown("### 💧 Humidity & Wind Speed Trend")
    items = forecast_data["list"][:14]
    times   = [datetime.fromtimestamp(i["dt"]) for i in items]
    humidity = [i["main"]["humidity"] for i in items]
    wind     = [i["wind"]["speed"] for i in items]

    bg   = "#0e1117" if theme == "dark" else "#ffffff"
    fc   = "#fafafa" if theme == "dark" else "#111111"
    grid = "#333333" if theme == "dark" else "#dddddd"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor(bg)

    for ax, data, color, label in [
        (ax1, humidity, "#00d4ff", "Humidity (%)"),
        (ax2, wind,     "#7b2ff7", "Wind Speed (m/s)"),
    ]:
        ax.set_facecolor(bg)
        ax.plot(times, data, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(times, data, alpha=0.15, color=color)
        ax.set_ylabel(label, color=fc, fontsize=10)
        ax.tick_params(colors=fc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, color=fc, fontsize=7)
        ax.grid(True, color=grid, alpha=0.4, linestyle="--")
        ax.spines[:].set_color(grid)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main() -> None:
    init_session_state()
    load_css(st.session_state.theme)

    # header
    st.markdown(
        """
        <div class="app-header">
            <span class="app-logo">🌤️</span>
            <span class="app-title">WeatherScope Pro</span>
            <span class="app-tagline">Real-Time Weather · ML Forecasting · Air Quality</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_key  = "d064e490cfc80a7025a018cabd648de5"   
    api      = WeatherAPI(api_key)
    history  = HistoryManager(Path("data/history.json"))

    city = render_sidebar(api, history)

    if city:
        with st.spinner(f"🌐 Fetching weather data for **{city}**..."):
            current  = api.get_current_weather(city)
            forecast = api.get_forecast(city)
            aqi      = api.get_aqi(city)

        if current.get("cod") != 200:
            st.error(f"❌ City **'{city}'** not found. Please check the spelling and try again.")
            return

        # persist
        st.session_state.current_data  = current
        st.session_state.forecast_data = forecast
        st.session_state.aqi_data      = aqi
        st.session_state.last_city     = city
        history.add(city)

    # ── render dashboard ──────────────────────────────────────
    if st.session_state.current_data:
        cur  = st.session_state.current_data
        fore = st.session_state.forecast_data
        aqi  = st.session_state.aqi_data
        unit = st.session_state.unit
        thm  = st.session_state.theme

        render_current_weather(cur, unit)
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📅 Forecast", "📈 Charts", "🤖 ML Prediction", "⚠️ Alerts & AQI"]
        )

        with tab1:
            render_forecast(fore, unit)

        with tab2:
            render_temperature_chart(fore, unit, thm)
            render_humidity_wind_chart(fore, thm)

        with tab3:
            render_ml_prediction(fore, unit, thm)

        with tab4:
            render_alerts(cur)
            st.markdown("---")
            render_aqi(aqi)

    else:
        # landing prompt
        st.markdown(
            """
            <div class="landing-section">
                <div class="landing-emoji">🌍</div>
                <h2 class="landing-title">Search any city to get started</h2>
                <p class="landing-sub">
                    Enter a city name in the sidebar to view real-time weather,
                    5-day forecasts, temperature trend charts, ML-powered predictions,
                    air quality index, and weather alerts.
                </p>
                <div class="landing-features">
                    <div class="feature-pill">🌡️ Real-Time Temp</div>
                    <div class="feature-pill">📅 5-Day Forecast</div>
                    <div class="feature-pill">🤖 ML Prediction</div>
                    <div class="feature-pill">🌬️ Air Quality</div>
                    <div class="feature-pill">⚠️ Weather Alerts</div>
                    <div class="feature-pill">📈 Trend Charts</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
