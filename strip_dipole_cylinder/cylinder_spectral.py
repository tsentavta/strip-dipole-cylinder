# -*- coding: utf-8 -*-
"""
Модуль 1: Спектральный аппарат цилиндра.

Реализует элемент матрицы поверхностных импедансов Z_{11}(n,h)
для бесконечного цилиндра радиуса a с диэлектрической проницаемостью
eps1 (внутри) и eps2 (снаружи).

Все формулы соответствуют главе 1 диссертации С.А. Шатрова (ПГУТИ, 2021).
Принято временное соглашение exp(i*omega*t).
"""

import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp


# Физические константы (СИ)
EPS0 = 8.8541878128e-12   # электрическая постоянная, Ф/м
MU0  = 1.25663706212e-6   # магнитная постоянная, Гн/м
C0   = 1.0 / np.sqrt(EPS0 * MU0)   # скорость света, м/с


def nu_m(h, k0, eps_m, mu_m):
    """
    Поперечное волновое число в среде m.
        nu_m = sqrt(h^2 - k0^2 * eps_m * mu_m).
    Соответствует определению из (1.14).

    Для случая h^2 < k0^2 * eps_m * mu_m возвращает чисто мнимое значение
    с правильной ветвью (отрицательная мнимая часть для радиационного
    условия, согласованного с exp(i*omega*t) и exp(-i*h*z)).
    """
    h_arr = np.asarray(h, dtype=complex)
    val = h_arr**2 - (k0**2) * eps_m * mu_m
    # малое регуляризующее смещение, чтобы избежать nu=0 точно
    if np.isscalar(val) or val.ndim == 0:
        if np.abs(val) < 1e-12 * (k0**2):
            val = val + 1e-12 * (k0**2)
    else:
        small = np.abs(val) < 1e-12 * (k0**2)
        val = np.where(small, val + 1e-12 * (k0**2), val)
    s = np.sqrt(val + 0j)
    # Стандартный выбор ветви: Re(s) >= 0;
    # если Re(s) < 0, перевернём знак.
    s = np.where(s.real < 0, -s, s)
    # Если чисто мнимое и Im > 0 — также перевернём (радиационная ветвь).
    s = np.where((np.abs(s.real) < 1e-14) & (s.imag > 0), -s, s)
    return s


def xi_zeta(n, h, a, k0, eps1, mu1, eps2, mu2):
    """
    Логарифмические производные функций Бесселя/Ханкеля
    в точке rho=a (формулы (1.42), (1.43) диссертации):

        xi   = J_n'(-i*nu1*a) / J_n(-i*nu1*a)
        zeta = H_n^{(2)'}(-i*nu2*a) / H_n^{(2)}(-i*nu2*a)

    Производные вычисляются по аргументу функций Бесселя/Ханкеля,
    то есть по rho. Используется правило: если arg = -i*nu*rho, то
    d/d(rho) [F(arg)] = (-i*nu) * F'(arg).

    В формуле же (1.42) под xi подразумевается отношение
    производной по аргументу функции к самой функции.
    """
    nu1 = nu_m(h, k0, eps1, mu1)
    nu2 = nu_m(h, k0, eps2, mu2)
    arg1 = -1j * nu1 * a
    arg2 = -1j * nu2 * a
    # jv(n, z) и jvp(n, z, 1) — функция и ее производная по z
    Jn   = jv(n, arg1)
    Jpn  = jvp(n, arg1, 1)
    Hn   = hankel2(n, arg2)
    Hpn  = h2vp(n, arg2, 1)
    # защита от деления на ноль
    Jn  = np.where(np.abs(Jn) < 1e-300, 1e-300+0j, Jn)
    Hn  = np.where(np.abs(Hn) < 1e-300, 1e-300+0j, Hn)
    xi   = Jpn  / Jn
    zeta = Hpn  / Hn
    return xi, zeta, nu1, nu2


def Z11(n, h, a, k0, eps1, mu1, eps2, mu2, omega=None):
    """
    Элемент Z_{11}(n,h) матрицы поверхностных импедансов
    цилиндра (формула (1.48) диссертации):

        Z_{11} = i*omega*mu0*mu1 * a^2 * nu1^2 * nu2^2 * (nu2^2*mu1*xi - nu1^2*mu2*zeta) /
                 [ k^2*a^2 * (nu2^2*eps1*xi - nu1^2*eps2*zeta) *
                            (nu2^2*mu1*xi  - nu1^2*mu2*zeta)
                   - (n*h)^2 * (nu2^2 - nu1^2)^2 ]

    omega — циклическая частота; если None, берётся omega = k0*c0.
    """
    if omega is None:
        omega = k0 * C0
    xi, zeta, nu1, nu2 = xi_zeta(n, h, a, k0, eps1, mu1, eps2, mu2)
    nu1sq = nu1**2
    nu2sq = nu2**2

    A = nu2sq * mu1 * xi - nu1sq * mu2 * zeta            # числитель
    B = nu2sq * eps1 * xi - nu1sq * eps2 * zeta          # сомножитель в знаменателе
    den = (k0**2) * (a**2) * B * A - (n * h)**2 * (nu2sq - nu1sq)**2

    num = 1j * omega * MU0 * mu1 * (a**2) * nu1sq * nu2sq * A
    return num / den


def Z11_asymptotic(h, k0, eps1, eps2, omega=None):
    """
    Асимптотика Z_{11} при |h|->infinity (формула (3.16)):
        Z_{11}^inf(h) = i*omega*mu0 / (k^2 * (eps1+eps2)) * |h|
    """
    if omega is None:
        omega = k0 * C0
    return 1j * omega * MU0 / (k0**2 * (eps1 + eps2)) * np.abs(h)


# ----------------------------------------------------------------------
# Простые проверочные расчёты
# ----------------------------------------------------------------------
if __name__ == "__main__":
    lam = 1.0
    k0 = 2 * np.pi / lam
    a = 0.1 * lam
    eps1 = 1.0
    eps2 = 1.0
    mu1 = 1.0
    mu2 = 1.0

    # Проверим Z11 при умеренных h (h порядка k0):
    for n in [0, 1, 2, 5]:
        for hl in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            h = hl * k0
            z = Z11(n, h, a, k0, eps1, mu1, eps2, mu2)
            zinf = Z11_asymptotic(h, k0, eps1, eps2)
            print(f"n={n:2d} h/k={hl:5.1f}  Z11={z.real:+.3e}{z.imag:+.3e}j   "
                  f"Z11_inf={zinf.imag:+.3e}j   ratio Im={z.imag/zinf.imag:.4f}")
        print()
