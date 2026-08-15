"""
app/components/sidebar.py
──────────────────────────
"""

import streamlit as st
from datetime import datetime
from app.components.ui import T

PAGES = [
    ("🏠", "Головна",     "Огляд та KPI"),
    ("🔮", "Прогноз",     "Прогнозування ризику"),
    ("🗺️", "Карта",       "Інтерактивна карта"),
    ("📊", "Аналітика",   "EDA та тренди"),
    ("⚖️", "Порівняння",  "4 моделі"),
    ("🧠", "SHAP / XAI",  "Пояснення моделі"),
    ("🎯", "Сценарії",    "Симуляція та звіти"),
    ("📁", "Датасет",     "Інформація про дані"),
    ("ℹ️", "Про проєкт",  "Технічний опис"),
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="padding:1.2rem 1rem .8rem;'
            'border-bottom:1px solid #1e2d3d">'
            '<span style="font-size:1.6rem;font-weight:700;'
            'color:#ff4d1a;display:block">FFIS</span>'
            '<span style="font-family:monospace;font-size:.58rem;'
            'text-transform:uppercase;letter-spacing:.18em;color:#4a5568">'
            'Forest Fire Intelligence</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if "page" not in st.session_state:
            st.session_state["page"] = "🏠 Головна"

        st.markdown(
            f'<p style="font-family:monospace;font-size:.6rem;text-transform:uppercase;'
            f'letter-spacing:.15em;color:{T["muted"]};margin:.5rem 0 .3rem">Регіон</p>',
            unsafe_allow_html=True,
        )
        region = st.radio(
            "Регіон",
            ["🇲🇦 Марокко", "🇩🇿 Алжир"],
            label_visibility="collapsed",
            key="region_selector",
        )
        st.session_state["selected_region"] = "Марокко" if "Марокко" in region else "Алжир"
        st.markdown("---")

        st.markdown(
            '<p style="font-family:monospace;font-size:.58rem;'
            'text-transform:uppercase;letter-spacing:.14em;color:#4a5568;'
            'margin:.8rem 0 .3rem .3rem">Навігація</p>',
            unsafe_allow_html=True,
        )

        for icon, name, hint in PAGES:
            full = f"{icon} {name}"
            if st.button(full, key=f"nav_{name}", help=hint,
                         use_container_width=True):
                st.session_state["page"] = full
                st.rerun()

        st.markdown("---")
        ml = st.session_state.get("model_loaded", False)
        dl = st.session_state.get("data_loaded",  False)

        def dot(ok: bool, label: str) -> str:
            color  = "#10b981" if ok else "#4a5568"
            shadow = f"box-shadow:0 0 5px {color}" if ok else ""
            return (
                f'<div style="display:flex;align-items:center;gap:.5rem;'
                f'font-size:.72rem;color:#8b9cb8;margin-bottom:.2rem">'
                f'<span style="width:6px;height:6px;border-radius:50%;'
                f'background:{color};{shadow}"></span>{label}</div>'
            )

        st.markdown(
            dot(ml, "ML Model")
            + dot(dl, "Dataset")
            + dot(True, "SHAP Engine")
            + dot(True, "Map"),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<p style="font-family:monospace;font-size:.58rem;'
            f'color:#4a5568;text-align:center;margin-top:.8rem">'
            f'{datetime.now().strftime("%Y-%m-%d  %H:%M")}</p>',
            unsafe_allow_html=True,
        )

    return st.session_state.get("page", "🏠 Головна")