import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# --- constants.py ---

# Fundamental constants from py_cea.f BLOCKDATA [cite: 1, 2]
# Using precise values from the source to maintain consistency with original solver
R_UNIV = 8314.51  # Universal gas constant, J/(kmol*K) [cite: 2]
PI = 3.14159265  # Pi [cite: 2]
AVOGADRO = 6.0221367e23  # Avogadro number [cite: 2]
BOLTZMANN = 1.380658e-23  # Boltzmann constant [cite: 2]


@dataclass
class Element:
    """
    Represents a chemical element.
    Data derived from BLOCKDATA atomic weights and symbols[cite: 2, 5].
    """
    symbol: str
    atomic_weight: float
    valency: float = 0.0  # From 'Valnce' data block [cite: 7]


class PeriodicTable:
    """
    Singleton class to hold atomic data consistent with CEA inputs.
    """

    def __init__(self):
        self.elements: Dict[str, Element] = {}
        self._load_data()

    def _load_data(self):
        # Atomic symbols from source [cite: 2, 3]
        symbols = [
            'H ', 'D ', 'HE', 'LI', 'BE', 'B ', 'C ', 'N ', 'O ', 'F ',
            'NE', 'NA', 'MG', 'AL', 'SI', 'P ', 'S ', 'CL', 'AR', 'K ', 'CA', 'SC',
            'TI', 'V ', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE', 'AS',
            'SE', 'BR', 'KR', 'RB', 'SR', 'Y ', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH',
            'PD', 'AG', 'CD', 'IN', 'SN', 'SB', 'TE', 'I ', 'XE', 'CS', 'BA', 'LA',
            'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM',
            'YB', 'LU', 'HF', 'TA', 'W ', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL',
            'PB', 'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'U ', 'NP',
            'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'UA'
        ]

        # Atomic weights from source [cite: 5, 6]
        weights = [
            1.00794, 2.014102, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067,
            15.9994, 18.9984032, 20.1797, 22.989770, 24.305, 26.981538, 28.0855, 30.973761,
            32.065, 35.453, 39.948, 39.0983, 40.078, 44.95591, 47.867, 50.9415, 51.9961,
            54.938049, 55.845, 58.933200, 58.6934, 63.546, 65.39, 69.723, 72.64,
            74.92160, 78.96, 79.904, 83.80, 85.4678, 87.62, 88.90585, 91.224, 92.90638,
            95.94, 97.9072, 101.07, 102.9055, 106.42, 107.8682, 112.411, 114.818,
            118.710, 121.760, 127.6, 126.90447, 131.293, 132.90545, 137.327, 138.9055,
            140.116, 140.90765, 144.9127, 145., 150.36, 151.964, 157.25, 158.92534,
            162.50, 164.93032, 167.259, 168.93421, 173.04, 174.967, 178.49, 180.9479,
            183.84, 186.207, 190.23, 192.217, 195.078, 196.96655, 200.59, 204.3833,
            207.2, 208.98038, 208.9824, 209.9871, 222.0176, 223.0197, 226.0254,
            227.0278, 232.0381, 231.03588, 238.02891, 237.0482, 244.0642, 243.0614,
            247.0703, 247.0703, 251.0587, 252.083, 17.03056  # Last one is pseudo-element (NH3?)
        ]

        # Valences from source [cite: 7]
        # Only loading first few for brevity in this example, full list should be populated
        valences = [
            1., 1., 0., 1., 2., 3., 4., 0., - 2., - 1., 0., 1., 2., 3., 4., 5.,
            4., - 1., 0., 1., 2., 3., 4., 5., 3., 2., 3., 2., 2., 2., 2., 3., 4., 3., 4.,
            - 1., 0., 1., 2., 3., 4., 5., 6., 7., 3., 3., 2., 1., 2., 3., 4., 3., 4.,
            - 1., 0., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            4., 5., 6., 7., 4., 4., 4., 3., 2., 1., 2., 3., 2., - 1., 0., 1., 2., 3., 4.,
            5., 6., 5., 4., 3., 3., 3., 3., 3., 0.
        ]

        # Populate dictionary
        for s, w, v in zip(symbols, weights, valences):
            clean_symbol = s.strip().upper()
            self.elements[clean_symbol] = Element(clean_symbol, w, v)

    def get_atomic_weight(self, symbol: str) -> float:
        symbol = symbol.strip().upper()
        if symbol in self.elements:
            return self.elements[symbol].atomic_weight
        else:
            raise ValueError(f"Unknown element: {symbol}")


# --- thermo.py (Part 1: Structure) ---

@dataclass
class NasaCoefficients:
    """
    Stores 7-term or 9-term polynomial coefficients for a temperature range.
    Corresponds to logic in subroutine CPHS.
    """
    t_min: float
    t_max: float
    coeffs: np.ndarray  # Array of 9 coefficients [a1...a9] from thermo.lib

    def calc_cp_h_s(self, T: float, R: float = R_UNIV):
        """
        Calculates Cp/R, H/RT, and S/R based on subroutine CPHS[cite: 29, 30].
        Note: Original Fortran uses specific indices for powers of T.

        Formulas derived from source CPHS[cite: 32, 33, 34]:
        Cp/R = a1/T^2 + a2/T + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
        H/RT = -a1/T^2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3 + a6*T^3/4 + a7*T^4/5 + a8/T
        S/R  = -a1/(2*T^2) - a2/T + a3*ln(T) + a4*T + a5*T^2/2 + a6*T^3/3 + a7*T^4/4 + a9
        """
        # Re-mapping coefficients to match typical standard NASA 9-coef format logic vs source.
        # In py_cea.f [cite: 30-34], the usage is quite specific about cx array.
        # We will need to verify the exact polynomial order against standard NASA 9-coef definitions.
        # Assuming standard NASA 9-coef for now:
        # a1, a2, a3, a4, a5, a6, a7 are for Cp
        # a8 is integration constant for H
        # a9 is integration constant for S

        T2 = T * T
        T3 = T * T2
        T4 = T2 * T2

        # Coefficients array a
        a = self.coeffs

        # Cp/R [cite: 34]
        cp_r = a[0] / T2 + a[1] / T + a[2] + a[3] * T + a[4] * T2 + a[5] * T3 + a[6] * T4

        # H/RT [cite: 32, 33] (Enthalpy)
        h_rt = -a[0] / T2 + a[1] * np.log(T) / T + a[2] + a[3] * T / 2.0 + a[4] * T2 / 3.0 + \
               a[5] * T3 / 4.0 + a[6] * T4 / 5.0 + a[7] / T

        # S/R [cite: 32, 33] (Entropy)
        s_r = -a[0] / (2.0 * T2) - a[1] / T + a[2] * np.log(T) + a[3] * T + a[4] * T2 / 2.0 + \
              a[5] * T3 / 3.0 + a[6] * T4 / 4.0 + a[8]

        return cp_r, h_rt, s_r