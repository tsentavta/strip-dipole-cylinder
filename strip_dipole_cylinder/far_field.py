# -*- coding: utf-8 -*-
"""
Модуль 3: Расчёт поля в дальней зоне.

Использует асимптотический метод стационарной фазы для перехода 
от спектрального представления (1.62) к дальнему полю.

ВЫВОД ФОРМУЛ ДАЛЬНЕЙ ЗОНЫ
==========================
Из главы 1.5, продольная компонента электрического поля во внешней 
области (m=2) имеет вид:

E_z^{(2)}(ρ,φ,z) = Σ_n e^{i n φ} ∫ Z_{11}(n,h) J_z^S(n,h) * 
                       H_n^{(2)}(-i nu_2 ρ) / H_n^{(2)}(-i nu_2 a) e^{-i h z} dh,
где J_z^S(n,h) = (Δ J_0(nΔ)/(4π)) * F(h),  F(h) = ∫_{-l}^l f(z') e^{i h z'} dz'.

ПЕРЕХОД В ДАЛЬНЮЮ ЗОНУ
======================
1) Сферические координаты: ρ = r sin θ, z = r cos θ.
2) Асимптотика H_n^{(2)}(z) при больших |z|:
     H_n^{(2)}(z) ~ sqrt(2/(π z)) e^{-i(z - n π/2 - π/4)}
3) Метод стационарной фазы по h:
   Φ(h) = -h z + sqrt(k_2² - h²) ρ ⇒ Φ'(h_s) = 0 ⇒ h_s = -k_2 cos θ.

ИТОГОВАЯ ФОРМУЛА:
   E_p(r,θ,φ) ≈ exp(-i k_2 r)/r * F_p(θ,φ).
   F_p(θ,φ) -- комплексная характеристика излучения.
   
Для расчёта НОРМИРОВАННОЙ ДН детальная нормировка несущественна, 
важна только относительная зависимость от θ и φ.
"""

import numpy as np
from scipy.special import jv as bessel_jv, jvp, hankel2, h2vp
from .cylinder_spectral import Z11, MU0, EPS0, C0, nu_m
from .sie_solver import f_spectrum


def far_field_components(theta, phi, c, k0, a, l, eps1, mu1, eps2, mu2, Delta,
                         r=1.0, Nmax_phi=20):
    """
    Вычисляет E_θ и E_φ в дальней зоне для направления (θ,φ).

    Возвращает: (E_theta, E_phi)  -- комплексные амплитуды.
    """
    k2 = k0 * np.sqrt(np.real(eps2) * np.real(mu2))
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Регуляризация около оси (θ→0 или π)
    sin_eff = sin_th if np.abs(sin_th) > 1e-8 else (1e-8 if sin_th >= 0 else -1e-8)

    h_s = -k2 * cos_th

    # nu_2 в точке стац. фазы
    # При |h_s| < k_2 → nu_2 чисто мнимое (распр. волна)
    nu2_hs = nu_m(h_s, k0, eps2, mu2)

    # Спектр тока в точке стац. фазы
    F_hs = f_spectrum(h_s, c, l)
    if not np.isfinite(np.abs(F_hs)):
        return 0j, 0j

    # Общий далекозонный префактор
    common_factor = (-1j * k2 / r) * np.exp(-1j * k2 * r)

    # Спектральные суммы
    Ez_spectral = 0.0 + 0.0j
    Erho_spectral = 0.0 + 0.0j
    Ephi_spectral = 0.0 + 0.0j

    arg_a = -1j * nu2_hs * a

    for n in range(-Nmax_phi, Nmax_phi + 1):
        Jn0 = bessel_jv(0, n * Delta)
        if np.abs(Jn0) < 1e-15:
            continue
        try:
            Z11_val = Z11(n, h_s, a, k0, eps1, mu1, eps2, mu2)
        except Exception:
            continue
        if not (np.isfinite(Z11_val.real) and np.isfinite(Z11_val.imag)):
            continue

        Hn_a = hankel2(n, arg_a)
        if np.abs(Hn_a) < 1e-300 or not np.isfinite(np.abs(Hn_a)):
            continue
        Hpn_a = h2vp(n, arg_a, 1)

        # Спектр тока
        Jz_S = (Delta * Jn0 / (4 * np.pi)) * F_hs

        # Фазовый множитель из метода стац. фазы
        phase = (1j) ** n * np.exp(1j * n * phi)

        # E_z в спектральном представлении
        contrib_Ez = phase * Z11_val * Jz_S / Hn_a
        # E_ρ: связана с производной по ρ функции α^{(2)}, 
        #      которая в дальней зоне даёт множитель -i*nu_2.
        contrib_Erho = phase * (h_s / nu2_hs) * Z11_val * Jz_S / Hn_a
        # E_φ: связана с производной по φ ⇒ множитель i*n
        contrib_Ephi = (phase * (1j * n) * Z11_val * Jz_S 
                        / (k2 * a * Hn_a + 1e-300))

        if all(np.isfinite([contrib_Ez.real, contrib_Ez.imag, 
                            contrib_Erho.real, contrib_Erho.imag,
                            contrib_Ephi.real, contrib_Ephi.imag])):
            Ez_spectral += contrib_Ez
            Erho_spectral += contrib_Erho
            Ephi_spectral += contrib_Ephi

    Ez_far = common_factor * sin_eff * Ez_spectral
    Erho_far = common_factor * sin_eff * Erho_spectral
    Ephi_far = common_factor * sin_eff * Ephi_spectral

    # Сферические компоненты
    E_theta = Erho_far * cos_th - Ez_far * sin_th
    E_phi = Ephi_far

    return E_theta, E_phi


def pattern_E_plane(c, k0, a, l, eps1, mu1, eps2, mu2, Delta,
                    N_theta=361, phi=0.0, Nmax_phi=20):
    """
    E-плоскость: фиксируем φ, варьируем θ ∈ [0, 2π].
    """
    theta_arr = np.linspace(0, 2 * np.pi, N_theta)
    intensity = np.zeros_like(theta_arr)
    for i, th in enumerate(theta_arr):
        Eth, Eph = far_field_components(th, phi, c, k0, a, l,
                                        eps1, mu1, eps2, mu2, Delta,
                                        Nmax_phi=Nmax_phi)
        I = np.abs(Eth) ** 2 + np.abs(Eph) ** 2
        if np.isfinite(I):
            intensity[i] = I
    Imax = np.max(intensity) if np.max(intensity) > 0 else 1.0
    return theta_arr, intensity / Imax, intensity


def pattern_H_plane(c, k0, a, l, eps1, mu1, eps2, mu2, Delta,
                    N_phi=361, theta=np.pi / 2, Nmax_phi=20):
    """
    H-плоскость: фиксируем θ=π/2, варьируем φ ∈ [0, 2π].
    """
    phi_arr = np.linspace(0, 2 * np.pi, N_phi)
    intensity = np.zeros_like(phi_arr)
    for i, ph in enumerate(phi_arr):
        Eth, Eph = far_field_components(theta, ph, c, k0, a, l,
                                        eps1, mu1, eps2, mu2, Delta,
                                        Nmax_phi=Nmax_phi)
        I = np.abs(Eth) ** 2 + np.abs(Eph) ** 2
        if np.isfinite(I):
            intensity[i] = I
    Imax = np.max(intensity) if np.max(intensity) > 0 else 1.0
    return phi_arr, intensity / Imax, intensity


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from .sie_solver import solve_current
    lam = 1.0
    k0 = 2 * np.pi / lam
    a = 0.1 * lam
    l = 0.25 * lam
    Delta = 0.1
    eps1 = 1.0; eps2 = 1.0; mu1 = mu2 = 1.0

    print("Решение СИУ...")
    c, M = solve_current(k0, a, l, eps1, mu1, eps2, mu2, Delta,
                         b=0.01 * l, E0=1.0, Kbasis=10, Nmax_phi=15,
                         Hmax_factor=30.0, n_quad_h=400, verbose=True)

    print("\nE-плоскость (φ=0):")
    theta_arr, F_E, _ = pattern_E_plane(c, k0, a, l, eps1, mu1, eps2, mu2,
                                        Delta, N_theta=73, Nmax_phi=12)
    for th, F in zip(theta_arr[::6], F_E[::6]):
        print(f"  θ={np.degrees(th):6.1f}° |F|² = {F:.4f}")

    print("\nH-плоскость (θ=π/2):")
    phi_arr, F_H, _ = pattern_H_plane(c, k0, a, l, eps1, mu1, eps2, mu2,
                                      Delta, N_phi=73, Nmax_phi=12)
    for ph, F in zip(phi_arr[::6], F_H[::6]):
        print(f"  φ={np.degrees(ph):6.1f}° |F|² = {F:.4f}")
