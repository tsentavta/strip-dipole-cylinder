# -*- coding: utf-8 -*-
"""
Базовые юнит-тесты для пакета strip_dipole_cylinder.

Запуск:
    pytest tests/
"""
import numpy as np
import pytest

from strip_dipole_cylinder import (
    Z11, nu_m, cheb_U, chi_k_spectral, 
    f_spectrum, compute_pattern, default_params
)


# ============================================================================
# Тесты модуля cylinder_spectral
# ============================================================================
class TestZ11:
    """Тесты функции Z₁₁(n,h) — спектрального импеданса цилиндра."""
    
    def test_z11_evenness_in_n(self):
        """Z₁₁(n,h) должна быть чётной по n: Z₁₁(-n,h) = Z₁₁(n,h)."""
        k0 = 2 * np.pi
        a = 0.1
        for n in [1, 2, 3, 5]:
            for h_factor in [0.5, 1.5, 3.0]:
                h = h_factor * k0
                z_pos = Z11(n, h, a, k0, 1.0, 1.0, 1.0, 1.0)
                z_neg = Z11(-n, h, a, k0, 1.0, 1.0, 1.0, 1.0)
                assert np.isclose(z_pos, z_neg, rtol=1e-10), \
                    f"Чётность по n нарушена при n={n}, h={h}"
    
    def test_z11_evenness_in_h(self):
        """Z₁₁(n,h) должна быть чётной по h: Z₁₁(n,-h) = Z₁₁(n,h)."""
        k0 = 2 * np.pi
        a = 0.1
        for n in [0, 1, 2]:
            for h_factor in [0.5, 1.5, 3.0]:
                h = h_factor * k0
                z_pos = Z11(n, h, a, k0, 1.0, 1.0, 1.0, 1.0)
                z_neg = Z11(n, -h, a, k0, 1.0, 1.0, 1.0, 1.0)
                assert np.isclose(z_pos, z_neg, rtol=1e-10), \
                    f"Чётность по h нарушена при n={n}, h={h}"
    
    def test_z11_finite(self):
        """Z₁₁ должна быть конечна для разумных значений параметров."""
        k0 = 2 * np.pi
        for h_factor in [0.1, 0.5, 1.5, 3.0, 10.0]:
            z = Z11(0, h_factor * k0, 0.1, k0, 2.0, 1.0, 1.0, 1.0)
            assert np.isfinite(z.real) and np.isfinite(z.imag), \
                f"Z11 не конечна при h/k0={h_factor}"


class TestNuM:
    """Тесты функции nu_m — поперечного волнового числа."""
    
    def test_nu_m_real_positive(self):
        """При |h| > k·sqrt(εμ) nu_m должна быть действительной положительной."""
        k0 = 2 * np.pi
        nu = nu_m(2 * k0, k0, 1.0, 1.0)
        assert nu.real > 0
        assert abs(nu.imag) < 1e-10
    
    def test_nu_m_imaginary(self):
        """При |h| < k·sqrt(εμ) nu_m должна быть мнимой."""
        k0 = 2 * np.pi
        nu = nu_m(0.5 * k0, k0, 1.0, 1.0)
        # nu² = h² - k² < 0  ⇒ nu чисто мнимое
        assert abs(nu.real) < 1e-6


# ============================================================================
# Тесты модуля sie_solver
# ============================================================================
class TestChebyshev:
    """Тесты полиномов Чебышёва."""
    
    def test_U_0(self):
        """U_0(t) = 1."""
        for t in [0.0, 0.5, -0.7]:
            assert np.isclose(cheb_U(0, t), 1.0)
    
    def test_U_1(self):
        """U_1(t) = 2t."""
        for t in [0.0, 0.3, -0.5]:
            assert np.isclose(cheb_U(1, t), 2 * t)


class TestChiKSpectral:
    """Тесты спектра базисных функций."""
    
    def test_chi_at_zero_k0(self):
        """chi_0(0) = π/2."""
        val = chi_k_spectral(0, 0.0)
        assert np.isclose(val.real, np.pi / 2, rtol=1e-6)
    
    def test_chi_at_zero_higher_k(self):
        """chi_k(0) = 0 для k >= 1."""
        for k in [1, 2, 3, 5]:
            val = chi_k_spectral(k, 0.0)
            assert np.isclose(abs(val), 0.0, atol=1e-10)
    
    def test_chi_scalar_or_array(self):
        """chi_k поддерживает скаляр и массив."""
        # Скаляр
        val_scalar = chi_k_spectral(0, 1.0)
        assert np.isscalar(val_scalar) or (hasattr(val_scalar, 'shape') and val_scalar.shape == ())
        # Массив
        val_arr = chi_k_spectral(0, np.array([0.5, 1.0, 2.0]))
        assert val_arr.shape == (3,)


# ============================================================================
# Тесты compute_pattern (интеграционные)
# ============================================================================
class TestComputePattern:
    """Интеграционные тесты полного расчёта."""
    
    @pytest.fixture
    def fast_params(self):
        """Быстрые параметры для тестов."""
        params = default_params()
        params['Kbasis'] = 8
        params['Nmax_phi'] = 10
        params['n_quad_h'] = 200
        params['N_theta'] = 73
        params['N_phi'] = 73
        return params
    
    def test_full_run(self, fast_params):
        """Полный расчёт должен завершиться без ошибок."""
        result = compute_pattern(fast_params, verbose=False)
        # Проверка структуры результата
        for key in ['c', 'M', 'z_arr', 'f_arr', 'Z_input',
                    'theta_E', 'pattern_E', 'phi_H', 'pattern_H', 'metrics']:
            assert key in result
    
    def test_current_boundary_condition(self, fast_params):
        """Распределение тока должно обращаться в ноль на концах."""
        result = compute_pattern(fast_params, verbose=False)
        f_arr = result['f_arr']
        # f должна стремиться к нулю на концах
        assert abs(f_arr[0]) < 1e-10
        assert abs(f_arr[-1]) < 1e-10
    
    def test_pattern_normalization(self, fast_params):
        """Нормированная ДН должна быть в [0, 1]."""
        result = compute_pattern(fast_params, verbose=False)
        assert np.all(result['pattern_E'] >= 0.0)
        assert np.all(result['pattern_E'] <= 1.0 + 1e-10)
        assert np.isclose(np.max(result['pattern_E']), 1.0, atol=1e-6)
    
    def test_metrics_present(self, fast_params):
        """Все метрики должны быть рассчитаны."""
        result = compute_pattern(fast_params, verbose=False)
        m = result['metrics']
        assert m['abs_z'] > 0
        assert isinstance(m['re_z'], float)
        assert isinstance(m['im_z'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
