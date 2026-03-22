"""
Monitor EEG Pro — Simulador Interactivo de Grado Médico
Versión 2.1 — Optimizado con Callbacks y st.rerun() nativo

Mejoras:
  - Arquitectura separada en clases (Simulator, UI, Renderer)
  - Burst Suppression con ciclo real y BSR calculado dinámicamente
  - Tarjeta de Insight con clasificación automática del estado clínico
  - Botones reactivos instantáneos usando callbacks (on_click)
  - Eliminación de bucles bloqueantes (while True)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import time

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor EEG Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados — UI oscura tipo monitor UCI
st.markdown("""
<style>
    .stApp { background-color: #070a0d; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    .stSlider > div > div > div { background: #1e2830; }
    .stExpander { background: #0d1620; border: 1px solid #1e2830 !important; }
    .stButton > button {
        background: #0d1620; color: #4a9eba;
        border: 1px solid #2a3840; font-family: monospace;
        letter-spacing: 1px; font-size: 12px;
    }
    .stButton > button:hover { background: #1a2830; color: #7dd4ee; }
    h1, h2, h3 { color: #4a9eba !important; font-family: monospace; letter-spacing: 2px; }
    .stMarkdown p { color: #6a8090; font-size: 12px; }
    div[data-testid="metric-container"] {
        background: #0d1620; border: 1px solid #1e2830;
        border-radius: 6px; padding: 8px 12px;
    }
    div[data-testid="metric-container"] label { color: #3a5060 !important; font-size: 10px; font-family: monospace; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #4a9eba !important; font-family: monospace; font-size: 22px;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')

# ─────────────────────────────────────────────
# CONSTANTES Y DEFINICIÓN DE BANDAS
# ─────────────────────────────────────────────
BAND_DEFS = [
    {"name": "SLOW",  "range": (0.1, 1.5),  "f_init": 0.8,  "a_init": 0.0,  "color": "#9370DB", "label_color": "#b090f0"},
    {"name": "DELTA", "range": (1.5, 4.0),  "f_init": 2.5,  "a_init": 0.0,  "color": "#e05050", "label_color": "#ff7070"},
    {"name": "THETA", "range": (4.0, 8.0),  "f_init": 6.0,  "a_init": 0.0,  "color": "#e09030", "label_color": "#ffb050"},
    {"name": "ALPHA", "range": (8.0, 13.0), "f_init": 10.0, "a_init": 0.0,  "color": "#30c070", "label_color": "#50e090"},
    {"name": "BETA",  "range": (14.0, 30.0),"f_init": 22.0, "a_init": 25.0, "color": "#3090c0", "label_color": "#50b0e0"},
    {"name": "GAMMA", "range": (30.0, 45.0),"f_init": 35.0, "a_init": 0.0,  "color": "#c040b0", "label_color": "#e060d0"},
]

CLINICAL_STATES = {
    "SLOW":  ("🔴 ESTADO DOWN",        "Silencio neuronal profundo. Aislamiento cortical completo.",    "#9370DB"),
    "DELTA": ("🟠 ANESTESIA PROFUNDA", "Delta dominante. Posible isquemia o sobredosis. Vigilar.",     "#e05050"),
    "THETA": ("🟡 TRANSICIÓN",         "Agentes volátiles activos. Transición hacia inconsciencia.",   "#e09030"),
    "ALPHA": ("🟢 ANESTESIA ESTABLE",  "Anteriorización Alpha (GABAérgico). Inconsciencia estable.",   "#30c070"),
    "BETA":  ("🔵 VIGILIA ACTIVA",     "Procesamiento cortical activo. Paciente despierto o sedación.", "#3090c0"),
    "GAMMA": ("🟣 AROUSAL / KETAMINA", "Alta tasa neuronal. Ketamina, dolor o respuesta a estímulo.", "#c040b0"),
}

FS = 128          
WIN_SEC = 8       
CHUNK = 16        
BUF_LEN = FS * WIN_SEC

# ─────────────────────────────────────────────
# CLASE SIMULADOR
# ─────────────────────────────────────────────
class EEGSimulator:
    def __init__(self):
        self.phases = np.zeros(len(BAND_DEFS))
        self.buffer = np.zeros(BUF_LEN, dtype=np.float32)
        self.ptr = 0          
        self.bs_clock = 0.0   

    def reset(self):
        self.buffer[:] = 0
        self.phases[:] = 0
        self.ptr = 0
        self.bs_clock = 0.0

    def get_ordered_buffer(self) -> np.ndarray:
        return np.concatenate([self.buffer[self.ptr:], self.buffer[:self.ptr]])

    def generate(self, freqs: list[float], amps: list[float], noise: float, bs_active: bool, bs_ratio: float) -> np.ndarray:
        t_arr = np.arange(CHUNK, dtype=np.float32) / FS
        chunk = np.zeros(CHUNK, dtype=np.float32)

        is_suppressed = False
        if bs_active:
            self.bs_clock = (self.bs_clock + CHUNK / FS) % 4.0
            is_suppressed = self.bs_clock < (4.0 * bs_ratio)

        for i, b in enumerate(BAND_DEFS):
            if amps[i] < 0.1:
                self.phases[i] = (self.phases[i] + 2 * np.pi * freqs[i] * CHUNK / FS) % (2 * np.pi)
                continue
            wave = amps[i] * np.sin(2 * np.pi * freqs[i] * t_arr + self.phases[i])
            self.phases[i] = (self.phases[i] + 2 * np.pi * freqs[i] * CHUNK / FS) % (2 * np.pi)
            chunk += wave.astype(np.float32)

        chunk += np.random.normal(0, max(noise, 0.01), CHUNK).astype(np.float32)

        if is_suppressed:
            chunk *= 0.04

        end = self.ptr + CHUNK
        if end <= BUF_LEN:
            self.buffer[self.ptr:end] = chunk
        else:
            split = BUF_LEN - self.ptr
            self.buffer[self.ptr:] = chunk[:split]
            self.buffer[:end - BUF_LEN] = chunk[split:]
        self.ptr = (self.ptr + CHUNK) % BUF_LEN

        return chunk

# ─────────────────────────────────────────────
# ANÁLISIS ESPECTRAL
# ─────────────────────────────────────────────
def compute_psd(buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(buf, FS, nperseg=256, noverlap=192, window='hann')
    return f, p

def compute_spectrogram(buf: np.ndarray) -> tuple:
    f, t, Sxx = signal.spectrogram(buf, FS, nperseg=128, noverlap=120, window='hann', scaling='density')
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db

def dominant_band(f: np.ndarray, psd: np.ndarray) -> tuple[str, float, float]:
    best_band, best_pow, best_freq = "BETA", 0.0, 20.0
    for b in BAND_DEFS:
        mask = (f >= b["range"][0]) & (f <= b["range"][1])
        if not mask.any(): continue
        band_pow = np.trapz(psd[mask], f[mask])
        if band_pow > best_pow:
            best_pow = band_pow
            best_band = b["name"]
            best_freq = f[mask][np.argmax(psd[mask])]
    return best_band, best_freq, best_pow

# ─────────────────────────────────────────────
# RENDERIZADO DE FIGURAS
# ─────────────────────────────────────────────
BG = "#070a0d"
GRID_COLOR = "#111820"

def fig_eeg(buf: np.ndarray, autoscale: bool = False, paused: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 2.6), facecolor=BG)
    face = "#0c1218" if paused else BG
    ax.set_facecolor(face)
    t = np.linspace(0, WIN_SEC, BUF_LEN)

    if paused:
        for y in np.arange(-300, 301, 10): ax.axhline(y, color="#1a2e3e", lw=0.3, zorder=0)
        for x in np.arange(0, WIN_SEC + 0.01, 0.2): ax.axvline(x, color="#1a2e3e", lw=0.3, zorder=0)
        for y in np.arange(-300, 301, 50): ax.axhline(y, color="#5a4010", lw=0.9, zorder=1)
        for x in np.arange(0, WIN_SEC + 0.01, 1.0): ax.axvline(x, color="#5a4010", lw=0.9, zorder=1)
        ax.axhline(0, color="#7a5818", lw=1.2, zorder=1)
    else:
        for y in np.arange(-300, 301, 50): ax.axhline(y, color=GRID_COLOR, lw=0.4, zorder=0)
        for x in np.arange(0, WIN_SEC + 0.01, 1.0): ax.axvline(x, color=GRID_COLOR, lw=0.4, zorder=0)

    y_lim = max(abs(buf).max() * 1.2, 30) if autoscale else 300
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlim(0, WIN_SEC)

    sig_lw = 1.4 if paused else 1.0
    ax.plot(t, buf, color="#22c55e", lw=sig_lw, zorder=2, antialiased=True)

    ax.annotate("50µV | 1s", xy=(0.12, -y_lim * 0.88), fontsize=8, color='#e0c020', fontfamily='monospace', fontweight='bold')

    title_str = "  EEG CORTICAL — CANAL SUMA   ■ PAUSADO" if paused else "  EEG CORTICAL — CANAL SUMA  "
    title_color = '#b07820' if paused else '#3a5060'

    ax.set_title(title_str, color=title_color, fontsize=8, loc='left', fontfamily='monospace')
    ax.set_ylabel("µV", color='#3a5060', fontsize=8)
    ax.set_xlabel("seg", color='#3a5060', fontsize=8)
    ax.tick_params(colors='#2a4050', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a3830' if paused else '#1e2830')

    fig.tight_layout(pad=0.4)
    return fig

def fig_dsa_psd(buf: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(12, 3.2), facecolor=BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.06)

    ax_dsa = fig.add_subplot(gs[0])
    ax_dsa.set_facecolor(BG)
    f_sp, t_sp, Sxx_db = compute_spectrogram(buf)
    mask_f = f_sp <= 45
    ax_dsa.pcolormesh(t_sp, f_sp[mask_f], Sxx_db[mask_f, :], cmap='jet', vmin=-5, vmax=30, shading='gouraud', rasterized=True)
    ax_dsa.set_ylim(0, 45)
    ax_dsa.set_xlim(t_sp[0], t_sp[-1])

    for b in BAND_DEFS:
        ax_dsa.axhline(b["range"][0], color=b["color"], lw=0.4, alpha=0.5, ls='--')
        ax_dsa.text(t_sp[-1] * 0.02, b["range"][0] + 0.3, b["name"], color=b["color"], fontsize=6, fontfamily='monospace', alpha=0.8)

    ax_dsa.set_title("DSA — Density Spectral Array", color='#3a5060', fontsize=8, fontfamily='monospace', loc='left')
    ax_dsa.set_ylabel("Hz", color='#3a5060', fontsize=8)
    ax_dsa.set_xlabel("seg", color='#3a5060', fontsize=8)
    ax_dsa.tick_params(colors='#2a4050', labelsize=7)
    for sp in ax_dsa.spines.values(): sp.set_edgecolor('#1e2830')

    ax_psd = fig.add_subplot(gs[1])
    ax_psd.set_facecolor(BG)
    f_w, psd = compute_psd(buf)
    mask_psd = f_w <= 45
    f_plot, p_plot = f_w[mask_psd], psd[mask_psd]
    peak_pow = p_plot.max() if p_plot.max() > 0 else 1.0
    y_max = peak_pow * 1.35

    for b in BAND_DEFS:
        ax_psd.axvspan(b["range"][0], b["range"][1], color=b["color"], alpha=0.12, zorder=0)
        mid = (b["range"][0] + b["range"][1]) / 2
        ax_psd.text(mid, y_max * 0.96, b["name"], color=b["label_color"], ha='center', fontsize=6, fontfamily='monospace', fontweight='bold', va='top')

    ax_psd.fill_between(f_plot, p_plot, color='#4a9eba', alpha=0.15, zorder=1)
    ax_psd.plot(f_plot, p_plot, color='#7de0f0', lw=1.8, zorder=2, antialiased=True)

    peak_idx = np.argmax(p_plot)
    ax_psd.axvline(f_plot[peak_idx], color='#ffff44', lw=0.8, ls='--', alpha=0.6)
    ax_psd.text(f_plot[peak_idx], peak_pow * 0.7, f"{f_plot[peak_idx]:.1f}Hz", color='#ffff44', fontsize=7, fontfamily='monospace', ha='center', bbox=dict(facecolor=BG, edgecolor='none', alpha=0.7))

    ax_psd.set_xlim(0, 45)
    ax_psd.set_ylim(0, y_max)
    ax_psd.set_title("PSD — Welch", color='#3a5060', fontsize=8, fontfamily='monospace', loc='left')
    ax_psd.set_xlabel("Hz", color='#3a5060', fontsize=8)
    ax_psd.tick_params(colors='#2a4050', labelsize=7)
    for sp in ax_psd.spines.values(): sp.set_edgecolor('#1e2830')

    fig.tight_layout(pad=0.4)
    return fig

def fig_channels(sim: 'EEGSimulator', freqs: list, amps: list, noise: float) -> plt.Figure:
    t = np.linspace(0, WIN_SEC, BUF_LEN)
    fig, axes = plt.subplots(len(BAND_DEFS), 1, figsize=(12, 7), facecolor=BG, sharex=True)
    plt.subplots_adjust(hspace=0.15, left=0.07, right=0.99, top=0.97, bottom=0.04)

    for i, (b, ax) in enumerate(zip(BAND_DEFS, axes)):
        ax.set_facecolor(BG)
        if amps[i] > 0.1:
            channel_sig = amps[i] * np.sin(2 * np.pi * freqs[i] * t) + np.random.normal(0, max(noise * 0.3, 0.1), BUF_LEN)
            ax.plot(t, channel_sig, color=b["color"], lw=0.9, antialiased=True)
            ax.set_ylim(-max(amps[i] * 1.5, 10), max(amps[i] * 1.5, 10))
        else:
            ax.plot(t, np.zeros(BUF_LEN), color=b["color"], lw=0.6, alpha=0.3)
            ax.set_ylim(-10, 10)
            ax.text(WIN_SEC / 2, 0, "— INACTIVO —", color=b["color"], alpha=0.3, ha='center', va='center', fontsize=8, fontfamily='monospace')

        ax.set_xlim(0, WIN_SEC)
        ax.set_ylabel(b["name"], color=b["color"], fontsize=8, fontfamily='monospace', fontweight='bold', rotation=0, labelpad=35, va='center')
        ax.grid(True, color=GRID_COLOR, lw=0.3, zorder=0)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2830')
        ax.tick_params(colors='#2a3840', labelsize=6)

    axes[-1].set_xlabel("seg", color='#3a5060', fontsize=8)
    return fig

# ─────────────────────────────────────────────
# INICIALIZACIÓN DE ESTADO
# ─────────────────────────────────────────────
def init_state():
    if "sim" not in st.session_state: st.session_state.sim = EEGSimulator()
    if "running" not in st.session_state: st.session_state.running = True
    if "show_channels" not in st.session_state: st.session_state.show_channels = False
    if "autoscale" not in st.session_state: st.session_state.autoscale = False
    if "freqs" not in st.session_state: st.session_state.freqs = [b["f_init"] for b in BAND_DEFS]
    if "amps" not in st.session_state: st.session_state.amps = [b["a_init"] for b in BAND_DEFS]
    if "noise" not in st.session_state: st.session_state.noise = 6.0
    if "bs_active" not in st.session_state: st.session_state.bs_active = False
    if "bs_ratio" not in st.session_state: st.session_state.bs_ratio = 0.5

# ─────────────────────────────────────────────
# SIDEBAR (CON CALLBACKS)
# ─────────────────────────────────────────────
def toggle_play():
    st.session_state.running = not st.session_state.running

def reset_sim():
    st.session_state.sim.reset()

def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("### ⬡ EEG PRO v2")

        col1, col2 = st.columns(2)
        with col1:
            lbl = "⏸ PAUSAR" if st.session_state.running else "▶ REANUDAR"
            st.button(lbl, on_click=toggle_play, use_container_width=True)
        with col2:
            st.button("↺ RESET", on_click=reset_sim, use_container_width=True)

        st.session_state.show_channels = st.toggle("Mostrar canales individuales", value=st.session_state.show_channels)
        st.session_state.autoscale = st.toggle("Autoescala EEG", value=st.session_state.autoscale)

        st.markdown("---")
        st.markdown("**BURST SUPPRESSION**")
        bs_active = st.toggle("Activar BS", value=st.session_state.bs_active, key="bs_toggle")
        bs_ratio = st.slider("BSR (fracción suprimida)", 0.0, 1.0, st.session_state.bs_ratio, 0.05, key="bs_ratio_slider")
        st.session_state.bs_active = bs_active
        st.session_state.bs_ratio = bs_ratio

        st.markdown("---")
        noise = st.slider("Ruido (µV)", 0.0, 30.0, st.session_state.noise, 0.5, key="noise_slider")
        st.session_state.noise = noise

        st.markdown("---")
        st.markdown("**BANDAS FRECUENCIALES**")
        freqs, amps = [], []
        for i, b in enumerate(BAND_DEFS):
            with st.expander(f"{'◼' if st.session_state.amps[i] > 0 else '◻'} {b['name']} ({b['range'][0]}–{b['range'][1]} Hz)"):
                f_val = st.slider("Frecuencia (Hz)", float(b["range"][0]), float(b["range"][1]), float(st.session_state.freqs[i]), 0.1, key=f"freq_{b['name']}")
                a_val = st.slider("Amplitud (µV)", 0.0, 150.0, float(st.session_state.amps[i]), 1.0, key=f"amp_{b['name']}")
                st.session_state.freqs[i] = f_val
                st.session_state.amps[i] = a_val
                freqs.append(f_val)
                amps.append(a_val)

        return freqs, amps, noise, bs_active, bs_ratio

# ─────────────────────────────────────────────
# PANEL DE INSIGHT CLÍNICO
# ─────────────────────────────────────────────
def render_insight(band: str, freq: float, total_amp: float, bs_active: bool, bs_ratio: float):
    state_label, state_desc, state_color = CLINICAL_STATES.get(
        band, ("— SIN SEÑAL —", "Amplitud insuficiente para análisis.", "#3a5060")
    )
    bsr_pct = f"{bs_ratio * 100:.0f}%" if bs_active else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BANDA DOMINANTE", band if total_amp > 0.5 else "—")
    c2.metric("FREQ. PICO", f"{freq:.1f} Hz" if total_amp > 0.5 else "—")
    c3.metric("AMP. TOTAL", f"{total_amp:.0f} µV" if total_amp > 0.5 else "—")
    c4.metric("BSR", bsr_pct)

    st.markdown(f"""
    <div style="
        background: #0d1620; border: 1px solid {state_color}44; border-left: 3px solid {state_color};
        border-radius: 4px; padding: 8px 14px; margin: 4px 0 8px 0;
        display: flex; align-items: center; gap: 12px;
    ">
        <div style="width: 10px; height: 10px; border-radius: 50%; background: {state_color}; box-shadow: 0 0 8px {state_color}88; flex-shrink: 0;"></div>
        <div>
            <span style="font-family: monospace; font-size: 13px; font-weight: 600; color: {state_color}; letter-spacing: 1px;">{state_label}</span>
            <span style="font-family: monospace; font-size: 11px; color: #6a8090; margin-left: 12px;">{state_desc}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BUCLE PRINCIPAL (SIN WHILE TRUE)
# ─────────────────────────────────────────────
def main():
    init_state()
    freqs, amps, noise, bs_active, bs_ratio = render_sidebar()
    st.markdown("<h1 style='font-size:1px;letter-spacing:3px;margin-bottom:1px:margin-top:0px'> </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:16px;letter-spacing:3px;margin-bottom:4px:margin-top:0px'>⬡MONITOR</h1>", unsafe_allow_html=True)

    sim: EEGSimulator = st.session_state.sim

    if st.session_state.running:
        sim.generate(freqs, amps, noise, bs_active, bs_ratio)

    buf = sim.get_ordered_buffer()
    total_amp = sum(amps)

    if total_amp > 0.1:
        f_w, psd = compute_psd(buf)
        mask45 = f_w <= 45
        dom_band, dom_freq, _ = dominant_band(f_w[mask45], psd[mask45])
    else:
        dom_band, dom_freq = "BETA", 0.0

    render_insight(dom_band, dom_freq, total_amp, bs_active, bs_ratio)

    fig = fig_eeg(buf, autoscale=st.session_state.autoscale, paused=not st.session_state.running)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig2 = fig_dsa_psd(buf)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    if st.session_state.show_channels:
        st.markdown("<p style='font-family:monospace;font-size:9px;letter-spacing:2px;color:#2a4050'>CANALES INDIVIDUALES</p>", unsafe_allow_html=True)
        fig3 = fig_channels(sim, freqs, amps, noise)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    if st.session_state.running:
        time.sleep(0.04)
        st.rerun()

if __name__ == "__main__":
    main()
