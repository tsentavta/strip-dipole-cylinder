# -*- coding: utf-8 -*-
"""
Модуль 2: Решение интегрального уравнения для тока в полосковом вибраторе
методом моментов с регуляризацией Карлемана-Векуа.

Базис: f(z) = sqrt(1 - (z/l)^2) * sum_k c_k * U_{k}(z/l),
где U_k - полиномы Чебышёва второго рода.
Этот базис автоматически удовлетворяет условию f(±l) = 0 и
правильному мейкснеровскому поведению на концах.

Аналитическая лемма (формула Гегенбауэра):
    ∫_{-1}^{1} sqrt(1-t^2) * U_k(t) * exp(i*x*t) dt
    = i^k * (k+1) * pi * J_{k+1}(x) / x
поэтому
    ∫_{-l}^{l} sqrt(1-(z/l)^2) U_k(z/l) e^{i*h*z'} dz'
    = l * pi * i^k * (k+1) * J_{k+1}(h*l) / (h*l)

Это позволяет аналитически взять интеграл по z' в (3.13) и свести задачу
к одномерному интегралу по h.
"""

import numpy as np
from scipy.special import jv as bessel_jv
from scipy.integrate import quad
from .cylinder_spectral import Z11, MU0, EPS0, C0


def cheb_U(k, t):
    """Полиномы Чебышёва второго рода U_k(t), |t|<=1."""
    th = np.arccos(np.clip(t, -1.0, 1.0))
    return np.sin((k + 1) * th) / np.where(np.abs(np.sin(th)) < 1e-300,
                                          1e-300, np.sin(th))


def chi_k_spectral(k, hl):
    """
    Спектр базисной функции psi_k(z) = sqrt(1-(z/l)^2) U_k(z/l) в нормировке:
        chi_k(h) = (1/l) ∫_{-l}^{l} psi_k(z) exp(i*h*z) dz / l   [безразм.]
    Явная формула:
        chi_k(h) = pi * i^k * (k+1) * J_{k+1}(h*l) / (h*l)
    """
    is_scalar = np.isscalar(hl) or (isinstance(hl, np.ndarray) and hl.ndim == 0)
    hl_arr = np.atleast_1d(hl).astype(float)
    out = np.zeros_like(hl_arr, dtype=complex)
    nz = np.abs(hl_arr) > 1e-12
    out[nz] = (np.pi * (1j) ** k * (k + 1)
               * bessel_jv(k + 1, hl_arr[nz]) / hl_arr[nz])
    if not np.all(nz):
        if k == 0:
            out[~nz] = np.pi * 0.5
        else:
            out[~nz] = 0.0
    if is_scalar:
        return complex(out[0])
    return out


def assemble_mom_matrix(k0, a, l, eps1, mu1, eps2, mu2, Delta,
                        Kbasis=12, Nmax_phi=20, Hmax_factor=40.0,
                        n_quad_h=600):
    """
    Собирает матрицу метода моментов для уравнения (3.13).

    Уравнение:  y(z) = ∫ f(z') K(z',z) dz',
    где K(z,z') = (Δ²/4) Σ_n J_0²(nΔ) ∫ Z11(n,h) exp(-ih(z-z')) dh.

    Раскладываем f(z') = sqrt(1-(z'/l)²) Σ_k c_k U_k(z'/l).
    Применяем метод Бубнова-Галёркина с тем же весовым базисом
    psi_m(z) = sqrt(1-(z/l)²) U_m(z/l).

    Имеем
        ∫ psi_m(z) ∫ f(z') K(z',z) dz' dz
        = (Δ²/4) Σ_n J_0²(nΔ) ∫ Z11(n,h) chi_m(h)* chi_k(h) dh,
    где chi_k(h) = pi (k+1) i^k J_{k+1}(hl)/(hl) - ВЕЩЕСТВЕННОЕ умноженное на i^k.

    После сборки получаем линейную систему M*c = b.

    Параметры:
      k0       : волновое число в свободном пр-ве
      a        : радиус цилиндра
      l        : полудлина вибратора
      eps1,mu1 : параметры среды внутри цилиндра
      eps2,mu2 : параметры среды снаружи
      Delta    : угловая полуширина полоски
      Kbasis   : число базисных функций (типично 8..16)
      Nmax_phi : усечение по азимутальным гармоникам |n|<=Nmax_phi
      Hmax_factor : предел интегрирования h в единицах k0 (h_max = Hmax_factor*k0)
      n_quad_h    : число точек квадратуры по h
    Возвращает:
      M : матрица (Kbasis, Kbasis) комплексная
    """
    # Сетка по h: 0 .. h_max (используем чётность по h => интегр от -inf до inf)
    h_max = Hmax_factor * k0
    # Гибридная сетка: больше точек около h=k0 (там полюсы Z_{11})
    # и более редкая в хвосте
    h_grid = np.concatenate([
        np.linspace(1e-6 * k0, k0 * 0.95, n_quad_h // 4),
        np.linspace(k0 * 0.96, k0 * 1.04, n_quad_h // 4),
        np.linspace(k0 * 1.05, 5 * k0, n_quad_h // 4),
        np.linspace(5 * k0, h_max, n_quad_h // 4),
    ])
    h_grid = np.unique(h_grid)
    h_grid = h_grid[h_grid > 0]
    # Веса трапеции
    dh = np.diff(h_grid)
    w_h = np.zeros_like(h_grid)
    w_h[1:] += dh / 2
    w_h[:-1] += dh / 2

    # Предвычислим chi_k(h) для всех k и h
    chi = np.zeros((Kbasis, len(h_grid)), dtype=complex)
    for k in range(Kbasis):
        chi[k, :] = chi_k_spectral(k, h_grid * l)

    # Предвычислим J_0(n*Delta)
    Jn0 = np.array([bessel_jv(0, n * Delta) for n in range(Nmax_phi + 1)])

    # Сборка матрицы M_{m,k}.
    # M_{m,k} = (Δ²/4) * Σ_{n=-Nmax}^{Nmax} J_0(nΔ)² * 
    #          ∫_{-inf}^{+inf} Z11(n,h) * chi_m(h)^* * chi_k(h) dh
    # Используем чётность Z11 и chi_k по h:
    #   Z11(n,h) = Z11(n,-h) (симметрия проверена)
    #   chi_k(h) при -h: J_{k+1}(-x) = (-1)^{k+1} J_{k+1}(x), 1/(-x) дает -1
    #   ⇒ chi_k(-h) = (-1)^{k+1+1} chi_k(h) = (-1)^k chi_k(h)
    # Поэтому chi_m(-h)^* * chi_k(-h) = (-1)^m (-1)^k chi_m(h)^* chi_k(h) = (-1)^{m+k} ...
    # Если m+k чётно, интеграл по [-inf, +inf] = 2*∫[0,+inf]; иначе = 0.
    # Это удобно — half of matrix is exactly zero (по чётности).
    M = np.zeros((Kbasis, Kbasis), dtype=complex)
    pref = (Delta**2) / 4.0

    # Цикл по n: используем чётность по n (J_0(nΔ)² и Z11 чётны по n)
    n_factor = np.zeros(Nmax_phi + 1)
    n_factor[0] = 1.0
    n_factor[1:] = 2.0   # учёт удвоения по симметрии n <-> -n

    for n_abs in range(Nmax_phi + 1):
        Jn0_sq = Jn0[n_abs]**2
        if Jn0_sq < 1e-25:
            continue
        # Вычисляем Z11(n_abs, h) на сетке
        Z_n = np.array([Z11(n_abs, h, a, k0, eps1, mu1, eps2, mu2)
                        for h in h_grid])
        for m in range(Kbasis):
            for k in range(Kbasis):
                if (m + k) % 2 != 0:
                    continue
                # Интеграл от -inf до +inf = 2 * Re part integration
                integrand = Z_n * np.conj(chi[m]) * chi[k]
                # Поскольку m+k чётно, integrand в +h и -h равны:
                contrib = 2.0 * np.sum(integrand * w_h)
                M[m, k] += pref * n_factor[n_abs] * Jn0_sq * contrib
    return M


def rhs_vector(k0, a, l, l0, b, Delta, E0, Kbasis):
    """
    Вектор правой части b_m = ∫ psi_m(z) y(z) dz, где
        y(z) = -E0 * pi * Delta * v(z),
        v(z) = (1/(2b)) * 1_{[l0-b, l0+b]}(z) - модель delta-зазора.
    Тогда b_m = -E0 * pi * Delta * (1/(2b)) * ∫_{l0-b}^{l0+b} psi_m(z) dz.
    """
    b_vec = np.zeros(Kbasis, dtype=complex)
    pref = -E0 * np.pi * Delta / (2 * b)
    # ∫_{l0-b}^{l0+b} sqrt(1-(z/l)²) U_m(z/l) dz
    # Используем замену t = z/l:
    # = l * ∫_{(l0-b)/l}^{(l0+b)/l} sqrt(1-t²) U_m(t) dt
    t1 = (l0 - b) / l
    t2 = (l0 + b) / l
    # Используем Гаусс-Лежандр на интервале [t1, t2]
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(40)
    # отображение [-1,1] -> [t1,t2]
    half = (t2 - t1) / 2
    cen = (t2 + t1) / 2
    t_nodes = cen + half * nodes
    sqrt_arr = np.sqrt(np.maximum(0.0, 1 - t_nodes**2))
    for m in range(Kbasis):
        Um = cheb_U(m, t_nodes)
        integ = np.sum(weights * sqrt_arr * Um) * half
        b_vec[m] = pref * l * integ
    return b_vec


def solve_current(k0, a, l, eps1, mu1, eps2, mu2,
                  Delta, l0=0.0, b=None, E0=1.0,
                  Kbasis=12, Nmax_phi=20,
                  Hmax_factor=40.0, n_quad_h=600,
                  verbose=False):
    """
    Решает интегральное уравнение и возвращает коэффициенты разложения c_k
    тока f(z) = sqrt(1-(z/l)²) Σ_k c_k U_k(z/l).
    """
    if b is None:
        b = 0.01 * l
    if verbose:
        print(f"  Сборка матрицы метода моментов: Kbasis={Kbasis}, "
              f"Nmax_phi={Nmax_phi}, h_max/k0={Hmax_factor}")
    M = assemble_mom_matrix(k0, a, l, eps1, mu1, eps2, mu2, Delta,
                            Kbasis=Kbasis, Nmax_phi=Nmax_phi,
                            Hmax_factor=Hmax_factor, n_quad_h=n_quad_h)
    rhs = rhs_vector(k0, a, l, l0, b, Delta, E0, Kbasis)
    # Регуляризация Тихонова: M_reg = M + alpha*I, alpha мала
    # для повышения устойчивости при плохой обусловленности
    alpha = 1e-9 * np.linalg.norm(M)
    M_reg = M + alpha * np.eye(Kbasis)
    c = np.linalg.solve(M_reg, rhs)
    return c, M


def f_of_z(z, c, l):
    """
    Восстанавливает функцию f(z) из коэффициентов c.
        f(z) = sqrt(1-(z/l)²) Σ_k c_k U_k(z/l)
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    t = z_arr / l
    mask = np.abs(t) <= 1.0
    out = np.zeros_like(z_arr, dtype=complex)
    sqrt_factor = np.sqrt(np.maximum(0.0, 1 - t[mask]**2))
    val = np.zeros(np.sum(mask), dtype=complex)
    for k, ck in enumerate(c):
        val += ck * cheb_U(k, t[mask])
    out[mask] = sqrt_factor * val
    return out if z_arr.shape else out[0]


def f_spectrum(h, c, l):
    """
    Спектр (фурье-образ) тока f(z).
        F(h) = ∫_{-l}^{l} f(z) exp(i*h*z) dz
             = l * Σ_k c_k * chi_k(h*l)
    """
    h_arr = np.atleast_1d(np.asarray(h, dtype=float))
    out = np.zeros_like(h_arr, dtype=complex)
    for k, ck in enumerate(c):
        out += ck * chi_k_spectral(k, h_arr * l)
    out = out * l
    if np.isscalar(h) or (isinstance(h, np.ndarray) and h.ndim == 0):
        return complex(out[0])
    return out


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Параметры из рис. 3.21 диссертации
    lam = 1.0
    k0 = 2 * np.pi / lam
    a = 0.1 * lam
    l = 0.25 * lam   # 2l = 0.5 λ — полуволновой вибратор
    Delta = 0.1
    eps1 = 1.0
    eps2 = 1.0
    mu1 = mu2 = 1.0
    b = 0.01 * l
    E0 = 1.0
    print(f"Решение СИУ при a/λ={a:.3f}, 2l/λ={2*l:.3f}, eps1={eps1}, Δ={Delta}")
    c, M = solve_current(k0, a, l, eps1, mu1, eps2, mu2, Delta,
                          b=b, E0=E0, Kbasis=10, Nmax_phi=15,
                          Hmax_factor=30.0, n_quad_h=400, verbose=True)
    print("Коэффициенты c_k разложения тока:")
    for k, ck in enumerate(c):
        print(f"  c_{k} = {ck:+.4e}")
    # Распределение тока вдоль вибратора
    z_arr = np.linspace(-l, l, 21)
    fz = f_of_z(z_arr, c, l)
    print("\nРаспределение f(z):")
    print("  z/l        Re f         Im f")
    for zi, fi in zip(z_arr, fz):
        print(f"  {zi/l:+.3f}    {fi.real:+.4e}    {fi.imag:+.4e}")
    print(f"\nf(0) = {fz[len(fz)//2]:+.4e}")
    # Входной импеданс: I_z(l0) = π Δ a f(l0); Z = U/I = 1/I (E0*2b нормировано)
    Iz0 = np.pi * Delta * a * fz[len(fz)//2]
    U = 2 * b * E0  # напряжение в зазоре
    Z_input = U / Iz0
    print(f"\nВходной импеданс: Z = {Z_input:+.3e} Ом")
    print(f"  Re Z = {Z_input.real:.3e}, Im Z = {Z_input.imag:.3e}")
