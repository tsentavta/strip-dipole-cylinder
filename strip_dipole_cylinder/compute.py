# -*- coding: utf-8 -*-
"""
Высокоуровневый интерфейс для расчёта характеристик полоскового вибраторного 
излучателя, расположенного на боковой поверхности диэлектрического цилиндра.

Содержит функцию compute_pattern(), выполняющую полный цикл расчёта 
(этапы 1-5 алгоритма) для заданного словаря параметров, и default_params() — 
набор параметров по умолчанию.
"""

import numpy as np

from .sie_solver import solve_current, f_of_z
from .far_field import pattern_E_plane, pattern_H_plane


def default_params():
    """
    Возвращает словарь параметров расчёта по умолчанию (соответствует
    рисунку 3.21 диссертации С.А. Шатрова).
    
    Returns
    -------
    dict
        Словарь со всеми параметрами задачи и численными настройками.
    """
    return {
        # Геометрия (длины в долях длины волны λ)
        'a_lambda':    0.1,    # радиус цилиндра, a/λ
        'l_lambda':    0.25,   # полудлина вибратора, l/λ (2l = 0.5λ → полуволновой)
        'Delta':       0.1,    # угловая полуширина полоски, рад
        'b_over_l':    0.01,   # ширина зазора возбуждения, b/l
        'l0_over_l':   0.0,    # положение точки питания, l₀/l (центр)
        
        # Электродинамические параметры
        'eps1':        1.0,    # ε внутри цилиндра
        'mu1':         1.0,    # μ внутри
        'eps2':        1.0,    # ε снаружи
        'mu2':         1.0,    # μ снаружи
        
        # Возбуждение
        'E0':          1.0,    # амплитуда стороннего поля, В/м
        
        # Численные параметры
        'Kbasis':      12,     # число базисных функций Чебышёва
        'Nmax_phi':    15,     # усечение по азимутальным гармоникам
        'Hmax_factor': 30.0,   # предел интегрирования h_max/k₀
        'n_quad_h':    400,    # число точек квадратуры по h
        
        # Параметры построения ДН
        'N_theta':     181,    # точек по θ (1° шаг)
        'N_phi':       181,    # точек по φ
        'phi_E':       0.0,    # значение φ для E-плоскости
        'theta_H':     np.pi / 2,  # значение θ для H-плоскости
    }


def compute_pattern(params=None, progress_callback=None, verbose=False):
    """
    Полный цикл расчёта характеристик полоскового вибратора.
    
    Выполняет 5 этапов алгоритма:
    1. Вычисление спектрального импеданса Z₁₁(n,h)
    2. Решение СИУ методом Бубнова-Галёркина
    3. Вычисление спектра распределения тока F̃(h)
    4. Расчёт поля в дальней зоне методом стационарной фазы
    5. Построение ДН в E- и H-плоскостях
    
    Parameters
    ----------
    params : dict, optional
        Словарь параметров (см. default_params()). Если не задан, 
        используются параметры по умолчанию.
    progress_callback : callable, optional
        Функция для отчёта о прогрессе. Вызывается с двумя аргументами:
        progress_callback(progress: float in [0, 1], message: str)
    verbose : bool, optional
        Если True, печатать отладочные сообщения в консоль.
    
    Returns
    -------
    dict
        Словарь с результатами:
        - 'params'    : использованные параметры
        - 'c'         : коэффициенты разложения тока (массив комплексных чисел)
        - 'M'         : матрица метода моментов
        - 'z_arr'     : сетка координат вдоль вибратора
        - 'f_arr'     : значения f(z) на сетке (комплексные)
        - 'Z_input'   : входной импеданс (комплексное число), Ом
        - 'theta_E'   : сетка углов θ для E-плоскости (рад)
        - 'pattern_E' : нормированная ДН в E-плоскости [0..1]
        - 'intens_E'  : абсолютные значения |E_θ|² + |E_φ|² 
        - 'phi_H'     : сетка углов φ для H-плоскости (рад)
        - 'pattern_H' : нормированная ДН в H-плоскости [0..1]
        - 'intens_H'  : абсолютные значения для H-плоскости
        - 'metrics'   : словарь интегральных характеристик 
                       (dtheta_3db, fb_ratio, knd, abs_z)
    """
    if params is None:
        params = default_params()
    
    # Распаковка параметров
    lam = 1.0
    k0 = 2 * np.pi / lam
    a = params['a_lambda'] * lam
    l = params['l_lambda'] * lam
    Delta = params['Delta']
    b = params['b_over_l'] * l
    l0 = params['l0_over_l'] * l
    eps1 = params['eps1']
    mu1 = params['mu1']
    eps2 = params['eps2']
    mu2 = params['mu2']
    E0 = params['E0']
    
    def _report(p, msg):
        if progress_callback is not None:
            progress_callback(p, msg)
        if verbose:
            print(f"[{p*100:5.1f}%] {msg}")
    
    # ===== Этапы 1-2: Решение СИУ =====
    _report(0.05, "Сборка матрицы метода моментов...")
    c, M = solve_current(
        k0, a, l, eps1, mu1, eps2, mu2, Delta,
        l0=l0, b=b, E0=E0,
        Kbasis=params['Kbasis'],
        Nmax_phi=params['Nmax_phi'],
        Hmax_factor=params['Hmax_factor'],
        n_quad_h=params['n_quad_h'],
        verbose=False
    )
    _report(0.40, "СИУ решено, восстанавливаем распределение тока...")
    
    # ===== Этап 3: Распределение тока =====
    z_arr = np.linspace(-l, l, 201)
    f_arr = f_of_z(z_arr, c, l)
    
    # Входной импеданс
    f_at_l0 = (np.interp(l0, z_arr, f_arr.real) + 
               1j * np.interp(l0, z_arr, f_arr.imag))
    Iz_l0 = np.pi * Delta * a * f_at_l0
    U = 2 * b * E0
    Z_input = U / Iz_l0 if np.abs(Iz_l0) > 1e-300 else 0.0 + 0.0j
    
    # ===== Этап 4-5: Расчёт ДН =====
    _report(0.45, "Расчёт ДН в E-плоскости...")
    theta_E, pattern_E, intens_E = pattern_E_plane(
        c, k0, a, l, eps1, mu1, eps2, mu2, Delta,
        N_theta=params['N_theta'], phi=params['phi_E'],
        Nmax_phi=params['Nmax_phi']
    )
    _report(0.75, "Расчёт ДН в H-плоскости...")
    phi_H, pattern_H, intens_H = pattern_H_plane(
        c, k0, a, l, eps1, mu1, eps2, mu2, Delta,
        N_phi=params['N_phi'], theta=params['theta_H'],
        Nmax_phi=params['Nmax_phi']
    )
    
    # ===== Расчёт интегральных характеристик =====
    _report(0.95, "Вычисление характеристик ДН...")
    metrics = compute_metrics(theta_E, pattern_E, phi_H, pattern_H, Z_input)
    
    _report(1.0, "Готово")
    
    return {
        'params':     params,
        'c':          c,
        'M':          M,
        'z_arr':      z_arr,
        'f_arr':      f_arr,
        'Z_input':    Z_input,
        'theta_E':    theta_E,
        'pattern_E':  pattern_E,
        'intens_E':   intens_E,
        'phi_H':      phi_H,
        'pattern_H':  pattern_H,
        'intens_H':   intens_H,
        'metrics':    metrics,
    }


def compute_metrics(theta_E, pattern_E, phi_H, pattern_H, Z_input):
    """
    Вычисление интегральных характеристик ДН.
    
    Returns
    -------
    dict
        - 'dtheta_3db' : ширина главного лепестка в E-плоскости по уровню −3 дБ, град
        - 'fb_ratio'   : передне-заднее отношение в H-плоскости, дБ
        - 'knd'        : коэффициент направленного действия (приближённо), дБ
        - 'abs_z'      : модуль входного импеданса, Ом
        - 're_z'       : действительная часть импеданса, Ом
        - 'im_z'       : мнимая часть импеданса, Ом
    """
    # Ширина главного лепестка в E-плоскости по уровню -3 дБ (= 0.5)
    # Берём значения в диапазоне θ ∈ [0, π], максимум должен быть около π/2
    mask = (theta_E >= 0) & (theta_E <= np.pi)
    theta_half = theta_E[mask]
    pat_half = pattern_E[mask]
    
    # Найти ширину по уровню 0.5
    if len(pat_half) > 0 and np.max(pat_half) > 0:
        idx_max = np.argmax(pat_half)
        # Ищем границы по уровню 0.5
        try:
            # справа от максимума
            right = pat_half[idx_max:]
            theta_right = theta_half[idx_max:]
            below = np.where(right < 0.5)[0]
            theta_right_3db = theta_right[below[0]] if len(below) > 0 else theta_right[-1]
            # слева
            left = pat_half[:idx_max + 1][::-1]
            theta_left = theta_half[:idx_max + 1][::-1]
            below = np.where(left < 0.5)[0]
            theta_left_3db = theta_left[below[0]] if len(below) > 0 else theta_left[-1]
            dtheta_3db_rad = theta_right_3db - theta_left_3db
            dtheta_3db = np.degrees(np.abs(dtheta_3db_rad))
        except Exception:
            dtheta_3db = float('nan')
    else:
        dtheta_3db = float('nan')
    
    # Передне-заднее отношение в H-плоскости
    # F = pattern_H в направлении φ=0; B = pattern_H в направлении φ=π
    idx_front = np.argmin(np.abs(phi_H - 0.0))
    idx_back = np.argmin(np.abs(phi_H - np.pi))
    front = pattern_H[idx_front] if pattern_H[idx_front] > 0 else 1e-10
    back = pattern_H[idx_back] if pattern_H[idx_back] > 0 else 1e-10
    if front > 0 and back > 0:
        fb_ratio = 10 * np.log10(front / back)
    else:
        fb_ratio = float('nan')
    
    # КНД (приближённая оценка через интеграл по углам)
    # КНД = 4π * F_max / ∫∫ F dΩ
    # Используем ДН только E-плоскости (предположение об азимутальной симметрии)
    # это очень грубая оценка
    try:
        # Интегрируем F sin θ по θ ∈ [0, π], предполагая независимость от φ
        sin_theta = np.sin(theta_half)
        # np.trapezoid в NumPy >= 2.0, np.trapz в старых версиях
        try:
            integral = np.trapezoid(pat_half * sin_theta, theta_half) * 2 * np.pi
        except AttributeError:
            integral = np.trapz(pat_half * sin_theta, theta_half) * 2 * np.pi
        if integral > 1e-10:
            knd = 10 * np.log10(4 * np.pi / integral)
            if not np.isfinite(knd):
                knd = None
        else:
            knd = None
    except Exception:
        knd = None
    
    abs_z = float(np.abs(Z_input))
    re_z = float(Z_input.real)
    im_z = float(Z_input.imag)
    
    return {
        'dtheta_3db': float(dtheta_3db) if dtheta_3db is not None and not np.isnan(dtheta_3db) else None,
        'fb_ratio':   float(fb_ratio) if not np.isnan(fb_ratio) else None,
        'knd':        knd,
        'abs_z':      abs_z,
        're_z':       re_z,
        'im_z':       im_z,
    }
