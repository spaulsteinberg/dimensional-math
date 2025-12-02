import numpy as np


def cp(usl: float, lsl: float, sigma: float):
    """
    Calculate CP
    :param usl: upper spec tolerance
    :param lsl: lower spec tolerance
    :param sigma: sigma
    :return: CP value
    """
    return (usl - lsl) / (6 * sigma)

def cpu(usl: float, mean: float, sigma: float):
    """
    Calculate CPU
    :param usl: upper spec tolerance
    :param mean: mean
    :param sigma: sigma
    :return: CPU value
    """
    return (usl - mean) / (3 * sigma)

def cpl(lsl: float, mean: float, sigma: float):
    """
    Calculate CPL
    :param lsl: lower spec tolerance
    :param mean: mean
    :param sigma: sigma
    :return: CPL value
    """
    return (mean - lsl) / (3 * sigma)

def cpk(usl: float, lsl: float, mean: float, sigma: float):
    """
    Calculate CPK
    :param usl: upper spec tolerance
    :param lsl: lower spec tolerance
    :param mean: mean
    :param sigma: sigma
    :return: CPK value
    """
    return min(cpu(usl, mean, sigma), cpl(lsl, mean, sigma))

def adsr_series(measurements: list[float], usl: float, lsl: float, s: float | None = 1) -> float | None:
    """
    Caculate ADSR over a series of measurements
    :param measurements: list of deviations
    :param usl: upper spec tolerance
    :param lsl: lower spec tolerance
    :param s: s value
    :return: ADSR value over a given series
    """
    num_measurements = len(measurements)
    if num_measurements == 0:
        return None
    sum = 0
    for measurement in measurements:
        b = 0 if usl >= measurement >= lsl else 1
        sum += ((max(measurement - usl, lsl - measurement) / (usl - lsl)) * b) * s
    return sum / num_measurements


def adsr_single(measurement: float, usl: float, lsl: float, s: float | None = 1) -> float:
    """
    Calculate ADSR for one measurement
    :param measurement: measurement
    :param usl: upper spec tolerance
    :param lsl: lower spec tolerance
    :param s: s value
    :return: ADSR value
    """
    b = 0 if usl >= measurement >= lsl else 1
    return ((max(measurement - usl, lsl - measurement) / (usl - lsl)) * b) * s


def polyfit(values: list[float], x: list[int]) -> tuple:
    """
    Calculate the polyfit line
    :param values: list of numbers
    :param x: x dimension
    :return: Tuple of the line and polyfit value
    """
    m, b = np.polyfit(x, values, 1)
    return m * np.array(x) + b, m


def range(values: list[float]) -> float:
    """
    Get the range for the set
    :param values: list of numbers
    :return: range of the set
    """
    return max(values) - min(values)
