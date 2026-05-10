"""
strip_dipole_cylinder — программный комплекс для электродинамического анализа
полосковых вибраторных излучателей на боковой поверхности диэлектрического цилиндра.

Реализует самосогласованный метод сингулярных интегральных уравнений 
с регуляризацией Карлемана-Векуа.

Пример использования:
    >>> from strip_dipole_cylinder import compute_pattern, default_params
    >>> params = default_params()
    >>> params['eps1'] = 3.5
    >>> result = compute_pattern(params)
    >>> # result содержит распределение тока, ДН в E- и H-плоскостях, импеданс
"""
from .cylinder_spectral import Z11, Z11_asymptotic, nu_m, xi_zeta, MU0, EPS0, C0
from .sie_solver import solve_current, f_of_z, f_spectrum, cheb_U, chi_k_spectral
from .far_field import far_field_components, pattern_E_plane, pattern_H_plane
from .compute import compute_pattern, default_params

__version__ = "1.0.0"
__author__ = "Кузнецов Е.М."
__email__ = "kuznetsov@example.com"

__all__ = [
    'compute_pattern',
    'default_params',
    'Z11',
    'Z11_asymptotic',
    'nu_m',
    'xi_zeta',
    'solve_current',
    'f_of_z',
    'f_spectrum',
    'cheb_U',
    'chi_k_spectral',
    'far_field_components',
    'pattern_E_plane',
    'pattern_H_plane',
    'MU0',
    'EPS0',
    'C0',
]
