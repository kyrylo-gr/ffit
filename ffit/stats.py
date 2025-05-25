import math
import numpy as np

_erfc_vec = np.vectorize(math.erfc)


def z_score(value_A, sigma_A, value_B, sigma_B):
    """Calculate z-score of agreement between two measurements.

    Calculate the standardized difference Z between two measurements:
    Z = |value_A - value_B| / sqrt(sigma_A^2 + sigma_B^2)

    Parameters:
        value_A (float): First measurement.
        sigma_A (float): Uncertainty of first measurement.
        value_B (float): Second measurement.
        sigma_B (float): Uncertainty of second measurement.

    Returns:
        float: The Z value quantifying the agreement.
    """
    # Compute the combined uncertainty
    denom = np.sqrt(sigma_A**2 + sigma_B**2)
    if denom == 0:
        raise ValueError("Combined uncertainty is zero; cannot compute Z.")

    # Return the standardized difference
    return np.where(denom == 0, np.nan, np.abs(value_A - value_B) / denom)


def p_value(value_A, sigma_A, value_B, sigma_B):
    """
    Calculate the two-tailed p-value for the difference between two measurements.

    Uses Z = |A - B| / sqrt(sigma_A^2 + sigma_B^2)
    and p = erfc(Z / sqrt(2)), vectorized via numpy.
    """
    Z = z_score(value_A, sigma_A, value_B, sigma_B)
    return _erfc_vec(Z / np.sqrt(2))
