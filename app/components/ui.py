"""
app/components/ui.py
─────────────────────
"""

from __future__ import annotations
import os
import streamlit as st
import plotly.graph_objects as go

T = {
    "bg":     "#111820",
    "border": "#1e2d3d",
    "fire":   "#ff4d1a",
    "fire_l": "#ff6b35",
    "safe":   "#10b981",
    "amber":  "#f59e0b",
    "blue":   "#60a5fa",
    "text":   "#e8f0f8",
    "sec":    "#8b9cb8",
    "muted":  "#4a5568",
}

PLOTLY = dict(
    paper_bgcolor=T["bg"],
    plot_bgcolor =T["bg"],
    font=dict(family="DM Sans, sans-serif", color=T["sec"], size=12),
    xaxis=dict(gridcolor=T["border"], tickfont=dict(color=T["muted"])),
    yaxis=dict(gridcolor=T["border"], tickfont=dict(color=T["muted"])),
    margin=dict(l=40, r=20, t=44, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["sec"])),
    colorway=[T["fire"], T["safe"], T["amber"], T["blue"]],
)

FIRE_CS = [
    [0.0,  "#10b981"],
    [0.35, "#84cc16"],
    [0.55, "#f59e0b"],
    [0.75, "#ff6b35"],
    [1.0,  "#ff4d1a"],
]


def inject_css() -> None:
    css_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "style.css"
    )
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def kpi(
    label:  str,
    value:  str,
    icon:   str = "📊",
    delta:  str = "",
    accent: str = "#ff4d1a",
) -> None:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi" style="border-top:2px solid {accent}">'
        f'<span class="kpi-icon">{icon}</span>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> None:
    icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔥"}
    st.markdown(
        f'<span class="risk-badge risk-{level}">'
        f'{icons.get(level, "⚪")} {level}</span>',
        unsafe_allow_html=True,
    )


def section(title: str, tag: str = "", icon: str = "") -> None:
    tag_html = (
        f'<span style="font-family:monospace;font-size:.6rem;'
        f'text-transform:uppercase;letter-spacing:.13em;color:{T["fire"]};'
        f'border:1px solid rgba(255,77,26,.3);border-radius:4px;padding:2px 6px">'
        f'{tag}</span>' if tag else ""
    )
    icon_html = f'<span style="font-size:1.1rem">{icon}</span>' if icon else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.6rem;'
        f'margin-bottom:.9rem;padding-bottom:.6rem;'
        f'border-bottom:1px solid {T["border"]}">'
        f'{icon_html}'
        f'<h2 style="font-size:1.45rem;font-weight:600;'
        f'color:{T["text"]};margin:0">{title}</h2>'
        f'{tag_html}</div>',
        unsafe_allow_html=True,
    )


def fire_bar(value: float) -> None:
    pct = round(value * 100)
    st.markdown(
        f'<div style="background:{T["border"]};border-radius:999px;'
        f'height:5px;margin:3px 0">'
        f'<div style="width:{pct}%;height:100%;border-radius:999px;'
        f'background:linear-gradient(90deg,{T["safe"]},{T["amber"]},{T["fire"]})">'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def gauge(value: float, title: str = "Fire Risk", height: int = 260) -> go.Figure:
    pct   = round(value * 100)
    color = (
        T["safe"]   if pct < 25 else
        T["amber"]  if pct < 55 else
        T["fire_l"] if pct < 75 else
        T["fire"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 38, "color": T["text"], "family": "sans-serif"},
        },
        title={
            "text": title.upper(),
            "font": {"size": 10, "color": T["muted"], "family": "monospace"},
        },
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=T["muted"], size=8)),
            bar=dict(color=color, thickness=0.22),
            bgcolor=T["border"],
            borderwidth=0,
            steps=[
                {"range": [0,   25], "color": "rgba(16,185,129,.06)"},
                {"range": [25,  55], "color": "rgba(245,158,11,.07)"},
                {"range": [55,  75], "color": "rgba(255,107,53,.09)"},
                {"range": [75, 100], "color": "rgba(255,77,26,.13)"},
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=T["bg"],
        height=height,
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(color=T["sec"]),
    )
    return fig


def apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        **PLOTLY,
        height=height,
        title=dict(
            text=title,
            font=dict(color=T["text"], size=13),
            x=0,
        ) if title else {},
    )
    return fig