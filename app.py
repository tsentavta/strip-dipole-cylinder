# -*- coding: utf-8 -*-
"""
Streamlit-приложение для расчёта характеристик полоскового вибраторного 
излучателя на боковой поверхности диэлектрического цилиндра.

Запуск:
    streamlit run app.py

Веб-интерфейс к программному комплексу strip_dipole_cylinder.

Автор: Кузнецов Е.М. (ПГУТИ, 2026)
Лицензия: MIT
"""

import io
import json
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from strip_dipole_cylinder import compute_pattern, default_params


# ============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================================
st.set_page_config(
    page_title="Полосковый вибратор на цилиндре",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Программный комплекс для электродинамического анализа "
                 "полосковых вибраторных излучателей на диэлектрическом цилиндре. "
                 "Магистерская диссертация Е.М. Кузнецова, ПГУТИ, 2026.",
    }
)

# Цветовая палитра «Midnight Executive»
COLOR_PRIMARY = '#1E2761'
COLOR_SECONDARY = '#3D52A1'
COLOR_LIGHT = '#CADCFC'
COLOR_ACCENT = '#D4AF37'

# Тонкая стилизация
st.markdown("""
<style>
    .main-header { color: #1E2761; font-weight: 700; }
    .metric-card { 
        background: #F5F7FB; 
        border-left: 4px solid #1E2761;
        padding: 12px 16px;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 20px; 
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .small-note { font-size: 12px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ПРЕСЕТЫ ПАРАМЕТРОВ
# ============================================================================
PRESETS = {
    "📋 Шатров (рис. 3.21)": {
        'a_lambda': 0.1, 'l_lambda': 0.25, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 1.0, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Полуволновой вибратор в свободном пространстве (свободно-стоящий случай)',
    },
    "🔵 Цилиндр с ε₁=2": {
        'a_lambda': 0.1, 'l_lambda': 0.25, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 2.0, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Слабодиэлектрический цилиндр (стекло, текстолит)',
    },
    "🟢 Цилиндр с ε₁=3.5": {
        'a_lambda': 0.1, 'l_lambda': 0.25, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 3.5, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Среднедиэлектрический цилиндр (керамика)',
    },
    "🟠 Цилиндр с ε₁=5": {
        'a_lambda': 0.1, 'l_lambda': 0.25, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 5.0, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Сильнодиэлектрический цилиндр — выраженная направленность',
    },
    "📐 Полноволновой (2l=λ)": {
        'a_lambda': 0.1, 'l_lambda': 0.5, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 1.0, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Полноволновой вибратор — отличие ДН от полуволнового',
    },
    "📐 Короткий (2l=λ/4)": {
        'a_lambda': 0.05, 'l_lambda': 0.125, 'Delta': 0.1,
        'b_over_l': 0.01, 'l0_over_l': 0.0,
        'eps1': 1.0, 'mu1': 1.0, 'eps2': 1.0, 'mu2': 1.0,
        'description': 'Электрически малый излучатель',
    },
}


# ============================================================================
# КЭШИРОВАНИЕ РАСЧЁТА
# ============================================================================
@st.cache_data(show_spinner=False, max_entries=20)
def cached_compute(params_tuple):
    """
    Кэшированный расчёт. Принимает tuple, чтобы быть хешируемым.
    """
    params = dict(params_tuple)
    return compute_pattern(params, verbose=False)


def hashable_params(params):
    """Преобразует словарь параметров в хешируемый tuple."""
    return tuple(sorted(params.items()))


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================================
def plot_polar_pattern(angles_rad, pattern, title="", color=COLOR_PRIMARY,
                       db_floor=-30, zero_loc='N', direction=-1):
    """
    Полярный график ДН в дБ.
    """
    angles_deg = np.degrees(angles_rad)
    pat_db = 10 * np.log10(np.maximum(pattern, 10**(db_floor / 10)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pat_db - db_floor,  # сдвигаем, чтобы radial axis был от 0 (= db_floor)
        theta=angles_deg,
        mode='lines',
        line=dict(color=color, width=2.5),
        hovertemplate='θ = %{theta:.1f}°<br>F = %{customdata:.2f} дБ<extra></extra>',
        customdata=pat_db,
    ))
    
    # Тики radial axis
    tick_levels = [-30, -20, -10, -3, 0]  # dB
    tick_vals = [t - db_floor for t in tick_levels]
    tick_text = [f"{t} дБ" for t in tick_levels]
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color=COLOR_PRIMARY)),
        polar=dict(
            radialaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[0, -db_floor],
                tickangle=90,
                gridcolor='#E5E7EB',
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                rotation=90 if zero_loc == 'N' else 0,
                direction='clockwise' if direction == -1 else 'counterclockwise',
                tickmode='array',
                tickvals=list(range(0, 360, 30)),
                ticktext=[f"{i}°" for i in range(0, 360, 30)],
                gridcolor='#E5E7EB',
                tickfont=dict(size=11),
            ),
            bgcolor='white',
        ),
        showlegend=False,
        height=400,
        margin=dict(t=50, b=10, l=30, r=30),
    )
    return fig


def plot_polar_pattern_multi(angles_rad_list, patterns_list, labels, colors,
                              title="", db_floor=-30, zero_loc='N', direction=-1):
    """
    Несколько полярных диаграмм на одних осях.
    """
    fig = go.Figure()
    for angles_rad, pattern, label, color in zip(angles_rad_list, patterns_list, labels, colors):
        angles_deg = np.degrees(angles_rad)
        pat_db = 10 * np.log10(np.maximum(pattern, 10**(db_floor / 10)))
        fig.add_trace(go.Scatterpolar(
            r=pat_db - db_floor,
            theta=angles_deg,
            mode='lines',
            line=dict(color=color, width=2),
            name=label,
            hovertemplate='θ = %{theta:.1f}°<br>F = %{customdata:.2f} дБ<extra>' + label + '</extra>',
            customdata=pat_db,
        ))
    
    tick_levels = [-30, -20, -10, -3, 0]
    tick_vals = [t - db_floor for t in tick_levels]
    tick_text = [f"{t} дБ" for t in tick_levels]
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color=COLOR_PRIMARY)),
        polar=dict(
            radialaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[0, -db_floor],
                tickangle=90,
                gridcolor='#E5E7EB',
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                rotation=90 if zero_loc == 'N' else 0,
                direction='clockwise' if direction == -1 else 'counterclockwise',
                tickmode='array',
                tickvals=list(range(0, 360, 30)),
                ticktext=[f"{i}°" for i in range(0, 360, 30)],
                gridcolor='#E5E7EB',
                tickfont=dict(size=11),
            ),
            bgcolor='white',
        ),
        showlegend=True,
        legend=dict(x=1.1, y=0.5, font=dict(size=11)),
        height=450,
        margin=dict(t=50, b=10, l=30, r=80),
    )
    return fig


def plot_current_distribution(z_arr, f_arr, l_lambda):
    """
    График распределения тока вдоль вибратора.
    """
    z_norm = z_arr / l_lambda
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_norm, y=f_arr.real,
        mode='lines',
        name='Re f(z)',
        line=dict(color='#0066CC', width=2.5),
        hovertemplate='z/λ = %{x:.3f}<br>Re f = %{y:.4e}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=z_norm, y=f_arr.imag,
        mode='lines',
        name='Im f(z)',
        line=dict(color='#CC0000', width=2, dash='dash'),
        hovertemplate='z/λ = %{x:.3f}<br>Im f = %{y:.4e}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text="Распределение поверхностной плотности тока",
                   x=0.5, font=dict(size=14, color=COLOR_PRIMARY)),
        xaxis=dict(title="z / λ", gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF'),
        yaxis=dict(title="f(z), А/м (произвольная нормировка)", gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF'),
        showlegend=True,
        legend=dict(x=0.7, y=0.95),
        height=400,
        plot_bgcolor='white',
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


# ============================================================================
# БОКОВАЯ ПАНЕЛЬ — ВВОД ПАРАМЕТРОВ
# ============================================================================
def render_sidebar():
    """Возвращает словарь параметров и режим (cloud/local)."""
    st.sidebar.markdown(f"""
    <div style='padding: 12px; background: {COLOR_PRIMARY}; border-radius: 6px;'>
        <div style='color: white; font-weight: 700; font-size: 16px;'>📡 Параметры расчёта</div>
        <div style='color: {COLOR_LIGHT}; font-size: 11px; margin-top: 4px;'>
            Полосковый вибратор на диэлектрическом цилиндре
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("")
    
    # ===== Пресеты =====
    preset_name = st.sidebar.selectbox(
        "📌 Пресет параметров",
        options=["✏️ Свои параметры"] + list(PRESETS.keys()),
        index=1,
        help="Готовые наборы параметров из научной литературы"
    )
    
    # Инициализация параметров из пресета
    if preset_name != "✏️ Свои параметры":
        preset = PRESETS[preset_name]
        st.sidebar.info(preset.get('description', ''))
        defaults = {k: v for k, v in preset.items() if k != 'description'}
    else:
        defaults = default_params()
    
    # ===== Геометрия =====
    with st.sidebar.expander("📐 Геометрия", expanded=True):
        a_lambda = st.slider(
            "Радиус цилиндра, a/λ",
            min_value=0.01, max_value=1.0, 
            value=float(defaults.get('a_lambda', 0.1)),
            step=0.01, format="%.2f",
            help="Электрический радиус ka = 2π · a/λ"
        )
        l_lambda = st.slider(
            "Полудлина вибратора, l/λ",
            min_value=0.05, max_value=1.0,
            value=float(defaults.get('l_lambda', 0.25)),
            step=0.01, format="%.2f",
            help="Полная длина 2l = 2 · l/λ. Полуволновой при l/λ = 0.25"
        )
        Delta = st.slider(
            "Угловая полуширина Δ, рад",
            min_value=0.01, max_value=0.5,
            value=float(defaults.get('Delta', 0.1)),
            step=0.01, format="%.2f"
        )
        b_over_l = st.slider(
            "Ширина зазора b/l",
            min_value=0.001, max_value=0.1,
            value=float(defaults.get('b_over_l', 0.01)),
            step=0.001, format="%.3f"
        )
        l0_over_l = st.slider(
            "Положение питания l₀/l",
            min_value=-0.5, max_value=0.5,
            value=float(defaults.get('l0_over_l', 0.0)),
            step=0.05, format="%.2f",
            help="0 — центральное возбуждение"
        )
    
    # ===== Электродинамические параметры =====
    with st.sidebar.expander("⚡ Среды", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Внутри**")
            eps1 = st.number_input("ε₁", min_value=1.0, max_value=20.0,
                                    value=float(defaults.get('eps1', 1.0)),
                                    step=0.1, format="%.2f")
            mu1 = st.number_input("μ₁", min_value=1.0, max_value=10.0,
                                   value=float(defaults.get('mu1', 1.0)),
                                   step=0.1, format="%.2f")
        with col2:
            st.markdown("**Снаружи**")
            eps2 = st.number_input("ε₂", min_value=1.0, max_value=20.0,
                                    value=float(defaults.get('eps2', 1.0)),
                                    step=0.1, format="%.2f")
            mu2 = st.number_input("μ₂", min_value=1.0, max_value=10.0,
                                   value=float(defaults.get('mu2', 1.0)),
                                   step=0.1, format="%.2f")
    
    # ===== Численные параметры =====
    with st.sidebar.expander("🔢 Численные параметры", expanded=False):
        # Режим производительности
        mode = st.radio(
            "Режим",
            options=["⚡ Быстрый", "⚖️ Сбалансированный", "🎯 Точный"],
            index=1,
            help="Влияет на размерности дискретизации"
        )
        
        if mode == "⚡ Быстрый":
            default_k, default_n, default_h, default_t = 8, 10, 200, 73
        elif mode == "⚖️ Сбалансированный":
            default_k, default_n, default_h, default_t = 12, 15, 400, 181
        else:
            default_k, default_n, default_h, default_t = 16, 20, 600, 361
        
        Kbasis = st.slider("Базисных функций K", 4, 24, default_k)
        Nmax_phi = st.slider("Азимут. гармоник N_max", 5, 30, default_n)
        n_quad_h = st.slider("Точек по h", 100, 800, default_h, step=50)
        N_theta = st.slider("Точек по θ (ДН)", 37, 361, default_t)
    
    # Кнопка сброса
    if st.sidebar.button("🔄 Сбросить к умолчаниям"):
        st.rerun()
    
    # ===== Информация =====
    st.sidebar.markdown("---")
    ka = 2 * np.pi * a_lambda
    st.sidebar.markdown(f"""
    <div class='small-note'>
        Электрический радиус: <b>ka = {ka:.3f}</b><br>
        Длина вибратора: <b>2l = {2*l_lambda:.2f}λ</b><br>
        Электр. длина: <b>k(2l) = {2*np.pi*2*l_lambda:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class='small-note' style='text-align: center;'>
        © Кузнецов Е.М., ПГУТИ, 2026<br>
        Магистерская диссертация<br>
        <a href='https://github.com/tsentavta/strip-dipole-cylinder'>GitHub</a> · 
        Версия 1.0
    </div>
    """, unsafe_allow_html=True)
    
    # Сборка словаря параметров
    params = {
        'a_lambda': a_lambda, 'l_lambda': l_lambda, 'Delta': Delta,
        'b_over_l': b_over_l, 'l0_over_l': l0_over_l,
        'eps1': eps1, 'mu1': mu1, 'eps2': eps2, 'mu2': mu2,
        'E0': 1.0,
        'Kbasis': Kbasis, 'Nmax_phi': Nmax_phi,
        'Hmax_factor': 30.0, 'n_quad_h': n_quad_h,
        'N_theta': N_theta, 'N_phi': N_theta,
        'phi_E': 0.0, 'theta_H': np.pi / 2,
    }
    return params


# ============================================================================
# ОСНОВНЫЕ ВКЛАДКИ ПРИЛОЖЕНИЯ
# ============================================================================
def tab_home():
    """Главная страница: краткое описание."""
    st.markdown("# 📡 Полосковый вибратор на диэлектрическом цилиндре")
    st.markdown(
        "**Программный комплекс для электродинамического анализа конформных "
        "микрополосковых антенн методом сингулярных интегральных уравнений**"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### О программе
        
        Это веб-приложение реализует расчёт характеристик излучения полоскового
        вибраторного излучателя длиной $2l$ и угловой ширины $2\\Delta$,
        расположенного на боковой поверхности бесконечного кругового цилиндра
        радиуса $a$ с диэлектрической проницаемостью $\\varepsilon_1$ внутри и
        $\\varepsilon_2$ снаружи.
        
        Программа решает **сингулярное интегральное уравнение** относительно 
        неизвестной поверхностной плотности тока методом Бубнова–Галёркина 
        с базисом полиномов Чебышёва второго рода с весом $\\sqrt{1-t^2}$, 
        что обеспечивает спектральную скорость сходимости.
        
        ### Что можно вычислить
        
        - 📊 **Распределение тока** $f_z(z)$ вдоль вибратора
        - 🎯 **Диаграмму направленности** в E- и H-плоскостях
        - ⚡ **Входное сопротивление** $Z = R + iX$
        - 📐 **Численные характеристики**: ширина главного лепестка, F/B, КНД
        - 🔬 **Параметрические исследования** при варьировании $\\varepsilon_1$ или $2l/\\lambda$
        
        ### Как пользоваться
        
        1. Задайте параметры в боковой панели слева (либо выберите готовый пресет)
        2. Перейдите на вкладку **«Расчёт»** и нажмите кнопку запуска
        3. Изучите результаты на вкладках **«ДН»**, **«Сравнение»** и **«Геометрия»**
        4. Экспортируйте данные через вкладку **«Экспорт»**
        """)
    
    with col2:
        st.markdown("### Реквизиты работы")
        st.info(
            """
            **Магистерская диссертация**
            
            **Кузнецов Е.М.**, группа ПИм-42
            
            *Прикладная информатика 09.04.03*
            
            Поволжский гос. университет 
            телекоммуникаций и информатики
            
            Научный руководитель:
            проф., д.ф.-м.н. **Клюев Д.С.**
            
            Самара, 2026
            """
        )
        
        st.markdown("### Технологии")
        st.markdown("""
        - 🐍 **Python 3.8+**
        - 🔢 **NumPy** (линейная алгебра)
        - 📐 **SciPy** (специальные функции)
        - 📊 **Plotly** (интерактивные графики)
        - 🌐 **Streamlit** (веб-интерфейс)
        """)
        
        st.markdown("### 📚 Ссылки")
        st.markdown("""
        - [📖 Документация](https://github.com/tsentavta/strip-dipole-cylinder)
        - [💻 Исходный код](https://github.com/tsentavta/strip-dipole-cylinder)
        - [🐛 Баги/предложения](https://github.com/tsentavta/strip-dipole-cylinder/issues)
        """)
    
    st.markdown("---")
    st.markdown("### 📐 Геометрия задачи")
    
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.markdown("""
        Структура состоит из:
        
        - **бесконечного кругового цилиндра** радиуса $a$
        - с проницаемостями $\\varepsilon_1, \\mu_1$ внутри и $\\varepsilon_2, \\mu_2$ снаружи
        - **идеально проводящей полоски** на боковой поверхности
        - длиной $2l$ и угловой ширины $2\\Delta$
        - возбуждаемой ЭДС в зазоре шириной $2b$
        """)
    with col_g2:
        st.latex(r"""
        \begin{aligned}
        \text{Граничное условие:}\quad E_z &= -E_z^{\text{ст}}(\varphi, z), \quad |z - l_0| \leq b \\
        E_z &= 0, \quad \text{на остальной части полоски}
        \end{aligned}
        """)
        st.latex(r"""
        Z_{11}(n,h) = \frac{i\omega\mu_0\mu_1 a^2 \nu_1^2 \nu_2^2 (\nu_2^2\mu_1\xi - \nu_1^2\mu_2\zeta)}{k^2 a^2 (\nu_2^2\varepsilon_1\xi - \nu_1^2\varepsilon_2\zeta)(\nu_2^2\mu_1\xi - \nu_1^2\mu_2\zeta) - (nh)^2(\nu_2^2-\nu_1^2)^2}
        """)


def tab_calc(params):
    """Вкладка с основным расчётом."""
    st.markdown("## 🎯 Расчёт характеристик излучателя")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button("🚀 Запустить расчёт", type="primary", use_container_width=True)
    with col_info:
        st.markdown(
            f"""<div class='small-note' style='padding-top: 8px;'>
            Текущие параметры: <b>a/λ = {params['a_lambda']:.2f}</b>, 
            <b>2l/λ = {2*params['l_lambda']:.2f}</b>, 
            <b>ε₁ = {params['eps1']:.1f}</b>, 
            <b>K = {params['Kbasis']}</b>, 
            <b>N_max = {params['Nmax_phi']}</b>
            </div>""", unsafe_allow_html=True
        )
    
    if not run_button and 'last_result' not in st.session_state:
        st.info("👈 Задайте параметры в боковой панели и нажмите «Запустить расчёт»")
        return
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(p, msg):
            progress_bar.progress(p)
            status_text.text(f"⏳ {msg}")
        
        start_time = time.time()
        try:
            result = cached_compute(hashable_params(params))
            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.success(f"✅ Готово за {elapsed:.1f} с")
            st.session_state['last_result'] = result
        except Exception as e:
            st.error(f"❌ Ошибка расчёта: {str(e)}")
            return
    
    # Используем результат из сессии (после расчёта или из кэша)
    if 'last_result' not in st.session_state:
        return
    result = st.session_state['last_result']
    metrics = result['metrics']
    
    # ===== Метрики =====
    st.markdown("### 📊 Основные характеристики")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if metrics['dtheta_3db'] is not None:
            st.metric("Δθ₃дБ (E-плоскость)", f"{metrics['dtheta_3db']:.1f}°",
                      help="Ширина главного лепестка по уровню −3 дБ")
        else:
            st.metric("Δθ₃дБ", "—")
    with m2:
        if metrics['fb_ratio'] is not None:
            st.metric("F/B (H-плоскость)", f"{metrics['fb_ratio']:.2f} дБ",
                      help="Передне-заднее отношение")
        else:
            st.metric("F/B", "—")
    with m3:
        if metrics['knd'] is not None:
            st.metric("КНД (оценка)", f"{metrics['knd']:.2f} дБ",
                      help="Коэффициент направленного действия")
        else:
            st.metric("КНД", "—")
    with m4:
        st.metric("|Z| (модуль)", f"{metrics['abs_z']:.1f} Ом",
                  delta=f"Re={metrics['re_z']:.0f}, Im={metrics['im_z']:.0f}",
                  delta_color="off",
                  help="Модуль входного импеданса")
    
    # ===== Графики =====
    st.markdown("### 📈 Графики")
    
    sub_dn1, sub_dn2 = st.columns(2)
    with sub_dn1:
        fig_E = plot_polar_pattern(
            result['theta_E'], result['pattern_E'],
            title=f"E-плоскость (φ = {np.degrees(params['phi_E']):.0f}°)",
            color=COLOR_PRIMARY, zero_loc='N', direction=-1
        )
        st.plotly_chart(fig_E, use_container_width=True)
    with sub_dn2:
        fig_H = plot_polar_pattern(
            result['phi_H'], result['pattern_H'],
            title=f"H-плоскость (θ = 90°)",
            color='#CC2936', zero_loc='E', direction=1
        )
        st.plotly_chart(fig_H, use_container_width=True)
    
    # Распределение тока
    fig_curr = plot_current_distribution(
        result['z_arr'], result['f_arr'], params['l_lambda']
    )
    st.plotly_chart(fig_curr, use_container_width=True)


def tab_compare():
    """Вкладка параметрического сравнения."""
    st.markdown("## 📊 Параметрическое сравнение")
    st.markdown(
        "Сравнение ДН для нескольких значений выбранного параметра. "
        "Остальные параметры берутся из боковой панели."
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        param_name = st.selectbox(
            "Варьируемый параметр",
            options=['eps1', 'l_lambda', 'a_lambda', 'Delta'],
            format_func=lambda x: {
                'eps1': 'Диэлектрическая проницаемость ε₁',
                'l_lambda': 'Полудлина вибратора l/λ',
                'a_lambda': 'Радиус цилиндра a/λ',
                'Delta': 'Угловая полуширина Δ'
            }.get(x, x)
        )
    
    with col2:
        if param_name == 'eps1':
            values_str = st.text_input("Значения через запятую", value="1, 2, 3.5, 5")
        elif param_name == 'l_lambda':
            values_str = st.text_input("Значения через запятую", value="0.15, 0.25, 0.5, 0.75")
        elif param_name == 'a_lambda':
            values_str = st.text_input("Значения через запятую", value="0.05, 0.1, 0.2, 0.4")
        else:  # Delta
            values_str = st.text_input("Значения через запятую", value="0.05, 0.1, 0.2, 0.4")
        
        try:
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
        except ValueError:
            st.error("Не могу распарсить значения. Используйте числа через запятую.")
            return
    
    # Берём базовые параметры из текущей сессии (sidebar)
    base_params = st.session_state.get('current_params', default_params())
    
    if st.button("🚀 Запустить сравнение", type="primary"):
        if len(values) > 6:
            st.warning("Слишком много значений (>6). Будут обработаны только первые 6.")
            values = values[:6]
        
        results = {}
        progress = st.progress(0)
        status = st.empty()
        
        for i, val in enumerate(values):
            status.text(f"⏳ Расчёт для {param_name} = {val} ({i+1}/{len(values)})")
            params_i = base_params.copy()
            params_i[param_name] = val
            try:
                results[val] = cached_compute(hashable_params(params_i))
            except Exception as e:
                st.error(f"Ошибка при {param_name} = {val}: {e}")
                continue
            progress.progress((i + 1) / len(values))
        
        status.success(f"✅ Расчёт завершён ({len(results)} вариантов)")
        st.session_state['compare_results'] = results
        st.session_state['compare_param'] = param_name
    
    # Отображение результатов
    if 'compare_results' not in st.session_state:
        st.info("👆 Выберите параметр, задайте значения и нажмите кнопку")
        return
    
    results = st.session_state['compare_results']
    param_name_disp = st.session_state.get('compare_param', 'param')
    
    # Палитра
    color_palette = ['#1E2761', '#3D8E5C', '#D4762B', '#CC2936', '#7B3F99', '#04827B']
    
    angles_E = [r['theta_E'] for r in results.values()]
    patterns_E = [r['pattern_E'] for r in results.values()]
    labels = [f"{param_name_disp} = {v}" for v in results.keys()]
    colors = color_palette[:len(results)]
    
    angles_H = [r['phi_H'] for r in results.values()]
    patterns_H = [r['pattern_H'] for r in results.values()]
    
    sub1, sub2 = st.columns(2)
    with sub1:
        fig_cE = plot_polar_pattern_multi(
            angles_E, patterns_E, labels, colors,
            title=f"E-плоскость", zero_loc='N', direction=-1
        )
        st.plotly_chart(fig_cE, use_container_width=True)
    with sub2:
        fig_cH = plot_polar_pattern_multi(
            angles_H, patterns_H, labels, colors,
            title=f"H-плоскость", zero_loc='E', direction=1
        )
        st.plotly_chart(fig_cH, use_container_width=True)
    
    # Таблица характеристик
    st.markdown("### 📋 Таблица характеристик")
    table_data = []
    for v, r in results.items():
        m = r['metrics']
        table_data.append({
            param_name_disp: v,
            "Δθ₃дБ, град": f"{m['dtheta_3db']:.1f}" if m['dtheta_3db'] else "—",
            "F/B, дБ":     f"{m['fb_ratio']:.2f}" if m['fb_ratio'] is not None else "—",
            "КНД, дБ":     f"{m['knd']:.2f}" if m['knd'] is not None else "—",
            "|Z|, Ом":     f"{m['abs_z']:.1f}",
            "Re Z, Ом":    f"{m['re_z']:.1f}",
            "Im Z, Ом":    f"{m['im_z']:.1f}",
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)


def tab_geometry(params):
    """Вкладка с 3D-визуализацией геометрии."""
    st.markdown("## 📐 Геометрия задачи")
    st.markdown("Интерактивная визуализация исследуемой структуры. Вращайте мышью.")
    
    a = params['a_lambda']
    l = params['l_lambda']
    Delta = params['Delta']
    b = params['b_over_l'] * l
    l0 = params['l0_over_l'] * l
    
    # Цилиндр (поверхность)
    z_cyl = np.linspace(-l * 1.4, l * 1.4, 30)
    phi_cyl = np.linspace(0, 2 * np.pi, 60)
    Z_cyl, PHI_cyl = np.meshgrid(z_cyl, phi_cyl)
    X_cyl = a * np.cos(PHI_cyl)
    Y_cyl = a * np.sin(PHI_cyl)
    
    fig = go.Figure()
    
    # Цилиндр (полупрозрачный)
    fig.add_trace(go.Surface(
        x=X_cyl, y=Y_cyl, z=Z_cyl,
        colorscale=[[0, 'rgb(220, 230, 255)'], [1, 'rgb(202, 220, 252)']],
        opacity=0.4,
        showscale=False,
        name='Цилиндр',
        hoverinfo='skip',
    ))
    
    # Полоска (с разрывом)
    n_phi_strip = 20
    n_z_strip = 60
    phi_strip = np.linspace(-Delta, Delta, n_phi_strip)
    
    # Сегменты полоски: верхняя часть и нижняя часть (с разрывом в зазоре)
    z_upper = np.linspace(l0 + b, l, n_z_strip // 2)
    z_lower = np.linspace(-l, l0 - b, n_z_strip // 2)
    
    for z_seg in [z_upper, z_lower]:
        Z_strip, PHI_strip = np.meshgrid(z_seg, phi_strip)
        X_strip = a * 1.005 * np.cos(PHI_strip)  # чуть снаружи
        Y_strip = a * 1.005 * np.sin(PHI_strip)
        fig.add_trace(go.Surface(
            x=X_strip, y=Y_strip, z=Z_strip,
            colorscale=[[0, 'rgb(30, 39, 97)'], [1, 'rgb(61, 82, 161)']],
            opacity=1.0,
            showscale=False,
            name='Полосковый вибратор',
            hoverinfo='skip',
        ))
    
    # Зазор возбуждения — красная подсветка
    if b > 0:
        z_gap = np.linspace(l0 - b, l0 + b, 5)
        Z_gap, PHI_gap = np.meshgrid(z_gap, phi_strip)
        X_gap = a * 1.005 * np.cos(PHI_gap)
        Y_gap = a * 1.005 * np.sin(PHI_gap)
        fig.add_trace(go.Surface(
            x=X_gap, y=Y_gap, z=Z_gap,
            colorscale=[[0, 'rgb(212, 175, 55)'], [1, 'rgb(212, 175, 55)']],
            opacity=1.0,
            showscale=False,
            name='Зазор возбуждения',
            hoverinfo='skip',
        ))
    
    # Ось z
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-l*1.5, l*1.5],
        mode='lines',
        line=dict(color='black', width=3),
        name='ось z',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[l * 1.6],
        mode='text',
        text=['z'],
        textfont=dict(size=18, color='black'),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x, λ', showgrid=True, zeroline=True),
            yaxis=dict(title='y, λ', showgrid=True, zeroline=True),
            zaxis=dict(title='z, λ', showgrid=True, zeroline=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.7)),
        ),
        height=600,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Параметры геометрии")
        st.markdown(f"""
        | Параметр | Значение |
        |----------|----------|
        | Радиус цилиндра, $a$ | {a:.3f} λ |
        | Длина излучателя, $2l$ | {2*l:.3f} λ |
        | Угловая ширина, $2\\Delta$ | {2*Delta:.3f} рад ({np.degrees(2*Delta):.1f}°) |
        | Зазор возбуждения, $2b$ | {2*b:.4f} λ |
        | Положение питания, $l_0$ | {l0:.3f} λ |
        """)
    with col2:
        st.markdown("### Электрические параметры")
        st.markdown(f"""
        | Параметр | Значение |
        |----------|----------|
        | Внутри цилиндра ($\\rho < a$): $\\varepsilon_1$ | {params['eps1']:.2f} |
        | Внутри цилиндра: $\\mu_1$ | {params['mu1']:.2f} |
        | Снаружи ($\\rho > a$): $\\varepsilon_2$ | {params['eps2']:.2f} |
        | Снаружи: $\\mu_2$ | {params['mu2']:.2f} |
        | Электрический радиус, $ka$ | {2*np.pi*a:.3f} |
        """)


def tab_export():
    """Вкладка экспорта результатов."""
    st.markdown("## 💾 Экспорт результатов")
    
    if 'last_result' not in st.session_state:
        st.warning("⚠️ Сначала запустите расчёт на вкладке «Расчёт»")
        return
    
    result = st.session_state['last_result']
    
    st.markdown("### 📥 Скачать данные")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV: ДН в E-плоскости
        csv_E = "theta_deg,F_normalized,F_dB\n"
        theta_deg = np.degrees(result['theta_E'])
        F_norm = result['pattern_E']
        F_dB = 10 * np.log10(np.maximum(F_norm, 1e-6))
        for t, f, f_db in zip(theta_deg, F_norm, F_dB):
            csv_E += f"{t:.4f},{f:.6e},{f_db:.4f}\n"
        st.download_button(
            "⬇️ ДН E-плоскость (CSV)",
            data=csv_E,
            file_name="pattern_E_plane.csv",
            mime="text/csv",
        )
    
    with col2:
        # CSV: ДН в H-плоскости
        csv_H = "phi_deg,F_normalized,F_dB\n"
        phi_deg = np.degrees(result['phi_H'])
        F_norm = result['pattern_H']
        F_dB = 10 * np.log10(np.maximum(F_norm, 1e-6))
        for p, f, f_db in zip(phi_deg, F_norm, F_dB):
            csv_H += f"{p:.4f},{f:.6e},{f_db:.4f}\n"
        st.download_button(
            "⬇️ ДН H-плоскость (CSV)",
            data=csv_H,
            file_name="pattern_H_plane.csv",
            mime="text/csv",
        )
    
    with col3:
        # CSV: распределение тока
        csv_I = "z_lambda,Re_f,Im_f,abs_f\n"
        z = result['z_arr'] / result['params']['l_lambda']
        f = result['f_arr']
        for zi, fi in zip(z, f):
            csv_I += f"{zi:.4f},{fi.real:.6e},{fi.imag:.6e},{abs(fi):.6e}\n"
        st.download_button(
            "⬇️ Ток f(z) (CSV)",
            data=csv_I,
            file_name="current_distribution.csv",
            mime="text/csv",
        )
    
    st.markdown("---")
    
    # Полный JSON со всеми результатами
    json_data = {
        'params': {k: v for k, v in result['params'].items() 
                   if not isinstance(v, np.ndarray)},
        'metrics': result['metrics'],
        'Z_input': {
            'real': result['Z_input'].real,
            'imag': result['Z_input'].imag,
            'abs': abs(result['Z_input']),
        },
        'pattern_E_plane': {
            'theta_deg': np.degrees(result['theta_E']).tolist(),
            'F_normalized': result['pattern_E'].tolist(),
        },
        'pattern_H_plane': {
            'phi_deg': np.degrees(result['phi_H']).tolist(),
            'F_normalized': result['pattern_H'].tolist(),
        },
        'current_distribution': {
            'z_lambda': (result['z_arr'] / result['params']['l_lambda']).tolist(),
            'Re_f': result['f_arr'].real.tolist(),
            'Im_f': result['f_arr'].imag.tolist(),
        },
    }
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    st.download_button(
        "📦 Полные результаты (JSON)",
        data=json_str,
        file_name="full_results.json",
        mime="application/json",
        use_container_width=True,
    )
    
    st.markdown("### 🔬 Параметры расчёта (для воспроизводимости)")
    st.json({k: v for k, v in result['params'].items() if not isinstance(v, np.ndarray)})


def tab_glossary():
    """Глоссарий терминов."""
    st.markdown("## 📚 Глоссарий")
    
    glossary = [
        ("**Конформная антенна**", 
         "антенна, повторяющая форму поверхности носителя (например, цилиндра)"),
        ("**Полосковый вибратор (ВИ)**",
         "узкая прямоугольная металлическая полоска, выполняющая функцию излучателя"),
        ("**Сингулярное интегральное уравнение (СИУ)**",
         "интегральное уравнение, ядро которого имеет неинтегрируемую особенность типа $1/(z-z')$ (Коши) или $\\ln|z-z'|$ (логарифмическую). СИУ с особенностью Коши является корректной по Адамару задачей."),
        ("**Метод Бубнова–Галёркина**",
         "проекционный метод решения операторных уравнений: неизвестная функция раскладывается по базису, а уравнение проектируется на тот же или другой базис, что приводит к системе линейных алгебраических уравнений"),
        ("**Полиномы Чебышёва**",
         "ортогональные на $[-1,1]$ полиномы. Первого рода $T_n(\\cos\\theta) = \\cos(n\\theta)$ ортогональны с весом $1/\\sqrt{1-t^2}$, второго рода $U_n$ — с весом $\\sqrt{1-t^2}$"),
        ("**Регуляризация Карлемана–Векуа**",
         "метод обращения интегрального уравнения с особенностью Коши, основанный на классической формуле обращения Карлемана"),
        ("**Условие Мейкснера**",
         "поведение электромагнитного поля вблизи рёбер идеально проводящего экрана: компоненты поля ведут себя как $r^{-1/2}$, где $r$ — расстояние до ребра"),
        ("**Метод стационарной фазы**",
         "асимптотический метод вычисления интегралов вида $\\int A(h) e^{i\\Phi(h)} dh$ при больших значениях параметра. Основной вклад в интеграл даёт окрестность точек, где $d\\Phi/dh = 0$"),
        ("**Диаграмма направленности (ДН)**",
         "угловая зависимость излучаемого поля антенны в дальней зоне. Часто представляется в нормированном виде $F(\\theta,\\varphi) = |E(\\theta,\\varphi)|^2/|E|^2_{\\max}$"),
        ("**E-плоскость**",
         "плоскость, проходящая через вектор электрического поля и направление главного максимума излучения. Для z-ориентированного вибратора — любая плоскость, содержащая ось $z$"),
        ("**H-плоскость**",
         "плоскость, перпендикулярная вектору электрического поля и проходящая через главный максимум. Для z-вибратора — плоскость $\\theta = \\pi/2$"),
        ("**Передне-заднее отношение (F/B)**",
         "отношение уровней излучения в направлении главного максимума к уровню в противоположном направлении, в дБ"),
        ("**КНД**",
         "коэффициент направленного действия, отношение плотности потока мощности в направлении максимума к среднему по сфере. КНД = $4\\pi/\\int F\\,d\\Omega$"),
        ("**Поверхностные волны**",
         "электромагнитные волны, локализованные вблизи направляющей структуры (например, диэлектрического цилиндра) и распространяющиеся вдоль неё"),
    ]
    
    for term, defn in glossary:
        with st.container():
            st.markdown(f"**{term}** — {defn}")
            st.markdown("")


def tab_about():
    """Информация о методе и литературе."""
    st.markdown("## 📖 О методе и литературе")
    
    st.markdown("""
    ### Историческая справка
    
    Систематическое исследование цилиндрических антенн ведёт свою историю с конца 
    XIX – середины XX века: уравнения Поклингтона (1897) и Халлена (1938), теория 
    Кинга, методы Уэйта, метод моментов Харрингтона.
    
    Применительно к **конформной геометрии** прорыв связан с работами 
    **самарской школы** (В.А. Неганов, Е.И. Нефёдов, Д.С. Клюев, А.Н. Дементьев, 
    Ю.В. Соколова): развит самосогласованный метод, основанный на сведении 
    электродинамической задачи к сингулярному интегральному уравнению с 
    особенностью Коши, корректному по Адамару.
    
    ### Ключевые формулы
    """)
    
    st.markdown("**Спектральный импеданс цилиндра** (формула 1.11 из диссертации):")
    st.latex(r"""
    Z_{11}(n,h) = \frac{i\omega\mu_0\mu_1 a^2 \nu_1^2 \nu_2^2 (\nu_2^2\mu_1\xi - \nu_1^2\mu_2\zeta)}{k^2 a^2 (\nu_2^2\varepsilon_1\xi - \nu_1^2\varepsilon_2\zeta)(\nu_2^2\mu_1\xi - \nu_1^2\mu_2\zeta) - (nh)^2(\nu_2^2-\nu_1^2)^2}
    """)
    
    st.markdown("где:")
    st.latex(r"""
    \nu_m = \sqrt{h^2 - k^2\varepsilon_m\mu_m}, \quad
    \xi = \frac{J_n'(-i\nu_1 a)}{J_n(-i\nu_1 a)}, \quad
    \zeta = \frac{H_n^{(2)\prime}(-i\nu_2 a)}{H_n^{(2)}(-i\nu_2 a)}
    """)
    
    st.markdown("**Сингулярное интегральное уравнение** (1.39):")
    st.latex(r"""
    \frac{1}{\pi}\int_{-l}^{l}\frac{f_z'(z')}{z'-z}\,dz' = G(z) + \int_{-l}^{l} f_z'(z') K_{\text{reg}}(z',z)\,dz'
    """)
    
    st.markdown("**Метод стационарной фазы для дальней зоны** (1.21):")
    st.latex(r"""
    h_s = -k_2 \cos\theta, \quad k_2 = k\sqrt{\varepsilon_2\mu_2}
    """)
    
    st.markdown("**Решение методом Бубнова–Галёркина** в базисе Чебышёва 2-го рода:")
    st.latex(r"""
    f_z(z) = \sqrt{1 - (z/l)^2}\,\sum_{k=0}^{K-1} c_k\,U_k(z/l)
    """)
    
    st.markdown("**Формула Гегенбауэра** (используется для аналитического интегрирования):")
    st.latex(r"""
    \int_{-1}^{1}\sqrt{1-t^2}\,U_k(t)\,e^{ixt}\,dt = i^k(k+1)\pi\frac{J_{k+1}(x)}{x}
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Ключевая литература")
    
    with st.expander("**Российская/советская школа**", expanded=False):
        st.markdown("""
        1. Мусхелишвили Н.И. *Сингулярные интегральные уравнения*. — М.: Наука, 1968.
        2. Гахов Ф.Д. *Краевые задачи*. — М.: Наука, 1977.
        3. Васильев Е.Н. *Возбуждение тел вращения*. — М.: Радио и связь, 1987.
        4. Лифанов И.К. *Метод сингулярных интегральных уравнений и численный 
        эксперимент*. — М.: ТОО «Янус», 1995.
        5. Захаров Е.В., Пименов Ю.В. *Численный анализ дифракции радиоволн*. — 
        М.: Радио и связь, 1982.
        6. Дементьев А.Н., Клюев Д.С., Неганов В.А., Соколова Ю.В. 
        *Сингулярные и гиперсингулярные интегральные уравнения в теории 
        зеркальных и полосковых антенн*. — М.: Радиотехника, 2015.
        7. Дементьев А.Н., Клюев Д.С., Соколова Ю.В. Сингулярное интегральное 
        уравнение для плотности тока конформного микрополоскового вибратора 
        на диэлектрическом цилиндре // Письма в ЖТФ. — 2017. — Т. 43, вып. 10.
        8. Клюев Д.С., Шатров С.А. (диссертационная работа, ПГУТИ, 2021).
        """)
    
    with st.expander("**Зарубежные источники**", expanded=False):
        st.markdown("""
        1. Hallén E. *Theoretical investigations into transmitting and receiving 
        qualities of antennas*. — Nova Acta Reg. Soc. Sci. Upsaliensis, 1938.
        2. King R.W.P. *Theory of Linear Antennas*. — Harvard Univ. Press, 1956.
        3. Harrington R.F. *Field Computation by Moment Methods*. — IEEE Press, 1968.
        4. Wait J.R. *Electromagnetic Wave Theory*. — Harper & Row, 1985.
        5. Volakis J.L., Sertel K. *Integral Equation Methods for Electromagnetics*. 
        — IET, 2012.
        6. Kress R. *Linear Integral Equations*. — Springer, 3rd ed., 2014.
        7. Meixner J. The behavior of electromagnetic fields at edges // 
        IEEE Trans. Antennas Propag., 1972, vol. 20, no. 4.
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Как процитировать эту работу")
    
    st.code("""
@mastersthesis{kuznetsov2026,
  author       = {Кузнецов, Е. М.},
  title        = {Разработка программного обеспечения для исследования 
                  полосковых вибраторных излучателей, расположенных 
                  на боковой поверхности диэлектрического цилиндра},
  type         = {Магистерская диссертация},
  school       = {Поволжский государственный университет 
                  телекоммуникаций и информатики},
  address      = {Самара},
  year         = {2026},
  url          = {https://github.com/tsentavta/strip-dipole-cylinder},
}
""", language="bibtex")


# ============================================================================
# MAIN
# ============================================================================
def main():
    params = render_sidebar()
    st.session_state['current_params'] = params
    
    # Создание вкладок
    tab_h, tab_c, tab_cmp, tab_g, tab_e, tab_gl, tab_a = st.tabs([
        "🏠 Главная",
        "🎯 Расчёт",
        "📊 Сравнение",
        "📐 Геометрия",
        "💾 Экспорт",
        "📚 Глоссарий",
        "📖 О методе"
    ])
    
    with tab_h:
        tab_home()
    with tab_c:
        tab_calc(params)
    with tab_cmp:
        tab_compare()
    with tab_g:
        tab_geometry(params)
    with tab_e:
        tab_export()
    with tab_gl:
        tab_glossary()
    with tab_a:
        tab_about()


if __name__ == "__main__":
    main()
