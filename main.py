"""
main.py
────────
Головний файл проєкту. Запуск:
    streamlit run main.py
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Це має бути ПЕРШИМ викликом Streamlit — нічого не ставити вище
st.set_page_config(
    page_title="FFIS — Forest Fire Intelligence System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.ui import inject_css
from app.components.sidebar import render_sidebar
from app.state.session import init_session
from app.pages.pages import (
    render_home,
    render_prediction,
    render_map,
    render_analytics,
    render_comparison,
    render_xai,
    render_scenarios,
    render_dataset,
    render_about,
)

# Завантажуємо CSS і ініціалізуємо сесію
inject_css()
init_session()

# Отримуємо вибрану сторінку з бічної панелі
page = render_sidebar()

# Дані і результати зі стану сесії
data    = st.session_state["app_data"]
results = st.session_state["model_results"]

# Маршрутизація сторінок
pages = {
    "🏠 Головна":     render_home,
    "🔮 Прогноз":     render_prediction,
    "🗺️ Карта":       render_map,
    "📊 Аналітика":   render_analytics,
    "⚖️ Порівняння":  render_comparison,
    "🧠 SHAP / XAI":  render_xai,
    "🎯 Сценарії":    render_scenarios,
    "📁 Датасет":     render_dataset,
    "ℹ️ Про проєкт":  render_about,
}

renderer = pages.get(page, render_home)
renderer(data, results)