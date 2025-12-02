import pytest
import numpy as np
from dp_math import *


def test_cp():
    assert cp(1.5, -1.5, 1) == 0.5

def test_cp_zero_div():
    with pytest.raises(ZeroDivisionError):
        cp(1, 1, 0)

def test_cpu():
    assert cpu(1.5, 1.5, 1) == 0
    assert cpu(2, 1, 1.5) == 1 / 4.5

def test_cpu_zero_div():
    with pytest.raises(ZeroDivisionError):
        cpu(1, 1, 0)

def test_cpl():
    assert cpl(1.5, 1.5, 1) == 0
    assert cpl(2, 1, 1.5) == -1 / 4.5

def test_cpl_zero_div():
    with pytest.raises(ZeroDivisionError):
        cpl(1, 1, 0)

def test_cpk():
    usl, lsl, mean, sigma = (1.5, -1.5, 1, 0.1)
    assert cpk(usl, lsl, mean, sigma) == min(cpu(usl, mean, sigma), cpl(lsl, mean, sigma))


def test_cpk_zero_div():
    with pytest.raises(ZeroDivisionError):
        cpk(1,1,1,0)

def test_aser_all_within_limits():
    # All measurements are between 10 and 20
    measurements = [12.0, 15.0, 18.0]
    usl, lsl = 20.0, 10.0
    # Expected result should be 0.0 since b=0 for all
    assert adsr_series(measurements, usl, lsl) == 0.0

def test_aser_one_above_usl():
    measurements = [10.0, 25.0, 15.0]
    usl, lsl = 20.0, 10.0
    s_factor = 1.0
    # Out-of-spec measurement: 25.0
    # Standardized error for 25: ((25 - 20) / (20 - 10)) * 1 = 5 / 10 = 0.5
    # Sum: 0.0 + 0.5 + 0.0 = 0.5
    # Average: 0.5 / 3 = 0.16666...
    expected = 0.5 / 3
    assert adsr_series(measurements, usl, lsl, s_factor) == pytest.approx(expected)

def test_aser_one_below_lsl_with_s_factor():
    measurements = [10.0, 5.0, 15.0]
    usl, lsl = 20.0, 10.0
    s_factor = 2.0
    # Out-of-spec measurement: 5.0
    # Standardized error for 5: ((10 - 5) / (20 - 10)) * 1 * 2.0 = (5 / 10) * 2 = 1.0
    # Sum: 0.0 + 1.0 + 0.0 = 1.0
    # Average: 1.0 / 3 = 0.33333...
    expected = 1.0 / 3
    assert adsr_series(measurements, usl, lsl, s_factor) == pytest.approx(expected)

def test_aser_empty_list_returns_none():
    assert adsr_series([], 20.0, 10.0) is None

def test_aser_measurement_on_limit():
    # Measurements exactly on the limits should be considered in spec (b=0)
    measurements = [10.0, 20.0]
    usl, lsl = 20.0, 10.0
    assert adsr_series(measurements, usl, lsl) == 0.0


def test_as_func_within_limits():
    # Measurement is between 10 and 20
    measurement, usl, lsl = 15.0, 20.0, 10.0
    assert adsr_single(measurement, usl, lsl) == 0.0

def test_as_func_above_usl():
    measurement, usl, lsl = 25.0, 20.0, 10.0
    s_factor = 1.0
    # Calculation: ((25 - 20) / (20 - 10)) * 1 = 5 / 10 = 0.5
    assert adsr_single(measurement, usl, lsl, s_factor) == pytest.approx(0.5)

def test_as_func_below_lsl_with_s_factor():
    measurement, usl, lsl = 5.0, 20.0, 10.0
    s_factor = 3.0
    # Calculation: ((10 - 5) / (20 - 10)) * 1 * 3 = (5 / 10) * 3 = 1.5
    assert adsr_single(measurement, usl, lsl, s_factor) == pytest.approx(1.5)

def test_as_func_on_usl():
    # On the limit should be 0 (in spec)
    measurement, usl, lsl = 20.0, 20.0, 10.0
    assert adsr_single(measurement, usl, lsl) == 0.0


@pytest.fixture
def linear_data():
    # Simple linear data: y = 2x + 1
    x = [0, 1, 2, 3]
    values = [1.0, 3.0, 5.0, 7.0]
    return values, x


def test_p_perfect_fit(linear_data):
    values, x = linear_data

    predicted_y, slope = polyfit(values, x)

    # Check the predicted values (should match the input values exactly)
    expected_y = np.array([1.0, 3.0, 5.0, 7.0])
    assert np.allclose(predicted_y, expected_y)

    # Check the slope (m)
    expected_slope = 2.0
    assert slope == pytest.approx(expected_slope)


def test_p_horizontal_line():
    # Data is constant: y = 5
    x = [1, 2, 3, 4]
    values = [5.0, 5.0, 5.0, 5.0]

    predicted_y, slope = polyfit(values, x)

    # Slope should be 0.0
    assert slope == pytest.approx(0.0)

    # Predicted values should all be 5.0
    expected_y = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.allclose(predicted_y, expected_y)


def test_p_negative_slope():
    # Data with a negative trend
    x = [0, 1, 2]
    values = [10, 5, 0]

    predicted_y, slope = polyfit(values, x)

    # Slope should be -5.0
    expected_slope = -5.0
    assert slope == pytest.approx(expected_slope)