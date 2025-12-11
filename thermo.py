"""
Thermodynamic properties calculator and database parser for NASA CEA formats.

This module handles:
1.  Storage of species data (composition, temperature intervals).
2.  Calculation of Cp, H, S, G using NASA-7 and NASA-9 polynomials.
3.  Parsing of 'thermo.inp' files in both legacy (NASA-7) and modern (NASA-9) formats.

References:
    NASA RP-1311 (CEA descriptions)
    NASA SP-273 (Original format)
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from constants import R_UNIV

@dataclass
class ThermoInterval:
    """
    Stores thermodynamic coefficients for a specific temperature range.

    Attributes:
        t_min (float): Minimum temperature for this interval.
        t_max (float): Maximum temperature for this interval.
        coeffs (np.ndarray): Polynomial coefficients.
            - Length 7 for NASA-7 format (Old CEA).
            - Length 9 for NASA-9 format (Modern CEA).
    """
    t_min: float
    t_max: float
    coeffs: np.ndarray


class Species:
    """
    Represents a chemical species with thermodynamic data.

    Attributes:
        name (str): The unique name of the species (e.g., "H2O").
        composition (Dict[str, float]): Elemental composition (e.g., {"H": 2, "O": 1}).
        molecular_weight (float): Molar mass in kg/kmol (or g/mol depending on DB units).
        intervals (List[ThermoInterval]): List of temperature intervals with coefficients.
    """

    def __init__(self, name: str, composition: Dict[str, float], molecular_weight: float = 0.0):
        self.name = name
        self.composition = composition
        self.molecular_weight = molecular_weight
        self.intervals: List[ThermoInterval] = []

    def add_interval(self, t_min: float, t_max: float, coeffs: Union[List[float], np.ndarray]):
        """Adds a temperature interval to the species."""
        self.intervals.append(ThermoInterval(t_min, t_max, np.array(coeffs, dtype=np.float64)))
        # Sort intervals by temperature to ensure correct lookup
        self.intervals.sort(key=lambda x: x.t_min)

    def get_properties(self, T: float) -> Tuple[float, float, float, float]:
        """
        Calculates dimensionless thermodynamic properties at temperature T.

        Args:
            T (float): Temperature in Kelvin.

        Returns:
            Tuple[float, float, float, float]: A tuple containing:
                - cp_r (Cp / R): Dimensionless heat capacity.
                - h_rt (H / RT): Dimensionless enthalpy.
                - s_r (S / R): Dimensionless entropy.
                - g_rt (G / RT): Dimensionless Gibbs free energy.

        Raises:
            ValueError: If no suitable interval is found for T (and extrapolation fails).
        """
        # 1. Find the correct interval
        interval = None
        for i in self.intervals:
            if i.t_min <= T <= i.t_max:
                interval = i
                break

        # Handle small numerical errors or extrapolation at boundaries
        if interval is None:
            if not self.intervals:
                raise ValueError(f"No thermo data available for {self.name}")

            # Simple extrapolation strategy: use the closest interval
            if T < self.intervals[0].t_min:
                interval = self.intervals[0]
            elif T > self.intervals[-1].t_max:
                interval = self.intervals[-1]
            else:
                # T is in a gap between intervals (rare in CEA, but possible)
                # Use the higher interval
                for i in self.intervals:
                    if T < i.t_max:
                        interval = i
                        break

        coeffs = interval.coeffs

        # 2. Select formula based on number of coefficients
        if len(coeffs) == 9:
            return self._calc_nasa9(T, coeffs)
        elif len(coeffs) == 7:
            return self._calc_nasa7(T, coeffs)
        else:
            raise ValueError(f"Unsupported coefficient count: {len(coeffs)} for {self.name}")

    def _calc_nasa9(self, T: float, c: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculates properties using the modern NASA-9 polynomial format.

        Formula (Standard NASA-9):
        Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
        H/RT = -a1*T^-2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3 + a6*T^3/4 + a7*T^4/5 + b1/T
        S/R  = -a1/(2*T^2) - a2/T + a3*ln(T) + a4*T + a5*T^2/2 + a6*T^3/3 + a7*T^4/4 + b2

        Coeffs array map: [a1, a2, a3, a4, a5, a6, a7, b1, b2]
        """
        t_inv2 = T ** -2
        t_inv = 1.0 / T
        t = T
        t2 = T ** 2
        t3 = T ** 3
        t4 = T ** 4
        ln_t = np.log(T)

        # Cp/R
        cp_r = (c[0] * t_inv2 + c[1] * t_inv + c[2] + c[3] * t +
                c[4] * t2 + c[5] * t3 + c[6] * t4)

        # H/RT
        h_rt = (-c[0] * t_inv2 + c[1] * t_inv * ln_t + c[2] + c[3] * t / 2.0 +
                c[4] * t2 / 3.0 + c[5] * t3 / 4.0 + c[6] * t4 / 5.0 + c[7] * t_inv)

        # S/R
        s_r = (-c[0] * t_inv2 / 2.0 - c[1] * t_inv + c[2] * ln_t + c[3] * t +
               c[4] * t2 / 2.0 + c[5] * t3 / 3.0 + c[6] * t4 / 4.0 + c[8])

        g_rt = h_rt - s_r
        return cp_r, h_rt, s_r, g_rt

    def _calc_nasa7(self, T: float, c: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculates properties using the legacy NASA-7 polynomial format.

        Formula (NASA-7):
        Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
        S/R  = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

        Coeffs array map: [a1, a2, a3, a4, a5, a6, a7]
        """
        t = T
        t2 = T ** 2
        t3 = T ** 3
        t4 = T ** 4
        ln_t = np.log(T)

        # Cp/R
        cp_r = c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4

        # H/RT
        h_rt = c[0] + c[1] * t / 2.0 + c[2] * t2 / 3.0 + c[3] * t3 / 4.0 + c[4] * t4 / 5.0 + c[5] / T

        # S/R
        s_r = c[0] * ln_t + c[1] * t + c[2] * t2 / 2.0 + c[3] * t3 / 3.0 + c[4] * t4 / 4.0 + c[6]

        g_rt = h_rt - s_r
        return cp_r, h_rt, s_r, g_rt


class ThermoDatabase:
    """
    Manages loading and accessing thermodynamic data for multiple species.
    Supports parsing both standard NASA-7 (thermo.inp) and NASA-9 formats.
    """

    def __init__(self):
        self.species: Dict[str, Species] = {}

    def load_from_file(self, filename: str):
        """
        Reads a thermo input file and populates the database.
        Automatically detects if a species entry is NASA-7 or NASA-9 format.

        Args:
            filename (str): Path to the thermo.inp file.
        """
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines, comments, or global headers
            if not line or line.startswith('!') or line.startswith('THERMO'):
                i += 1
                continue

            # Stop at END
            if line.upper().startswith('END'):
                break

            # Heuristic to detect format start
            # NASA-9 usually has a name line, then a composition line starting with ' 1' or similar codes.
            # NASA-7 has a fixed 4-line structure where line 1 has T ranges at the end.

            # Let's peek at the current block to decide.
            header_line = lines[i]

            # Check for NASA-9 signature (often has many words, date, but T ranges are NOT at cols 45-75 usually)
            # A strong indicator for NASA-7 is numbers at cols 45-75 on the FIRST line.
            is_nasa7 = False
            try:
                # NASA-7 puts temperatures here:
                # t_low (45:55), t_high (55:65), t_mid (65:75) - strictly formatted
                if len(header_line) > 65:
                    _ = float(header_line[45:55])
                    _ = float(header_line[55:65])
                    # If these parse as floats, it's highly likely NASA-7
                    is_nasa7 = True
            except ValueError:
                is_nasa7 = False

            if is_nasa7:
                i = self._parse_species_nasa7(lines, i)
            else:
                # Assume NASA-9 or try to parse as such
                # NASA-9 blocks are variable length, so the parser must return the new index
                try:
                    i = self._parse_species_nasa9(lines, i)
                except Exception as e:
                    print(f"Warning: Failed to parse block at line {i}: {e}. Skipping line.")
                    i += 1

    def _parse_species_nasa7(self, lines: List[str], idx: int) -> int:
        """
        Parses a single species block in NASA-7 format (4 lines).
        Returns the index of the next line to process.
        """
        line1 = lines[idx]

        # 1. Parse Name (cols 0-18)
        name = line1[0:18].split()[0].strip()

        # 2. Parse Composition (cols 24-44)
        # 4 sets of (Symbol: 2 chars, Count: 3 chars float-like)
        composition = {}
        comp_segment = line1[24:44]
        for k in range(0, 20, 5):
            chunk = comp_segment[k:k + 5]
            if len(chunk) < 5: continue
            sym = chunk[0:2].strip()
            qty_str = chunk[2:5].strip()
            if sym and qty_str:
                try:
                    qty = float(qty_str)
                    if qty > 0:
                        composition[sym] = qty
                except ValueError:
                    pass

        # 3. Parse Temperatures (cols 45-75)
        # t_low, t_high, t_mid
        try:
            t_low = float(line1[45:55])
            t_high = float(line1[55:65])
            t_mid = float(line1[65:75])
        except ValueError:
            # Fallback if corrupt, though is_nasa7 check passed
            return idx + 1

        current_species = Species(name, composition)

        # 4. Parse Coefficients (Lines 2, 3, 4)
        # Total 3 lines. 5 coeffs per line (width 15).
        # Logic: First 7 coeffs -> High Temp (T_mid to T_high)
        #        Next 7 coeffs -> Low Temp (T_low to T_mid)
        raw_coeffs = []
        for offset in range(1, 4):
            if idx + offset >= len(lines): break
            cline = lines[idx + offset]
            for pos in range(0, 75, 15):
                if pos + 15 > len(cline): break  # End of line
                val_str = cline[pos:pos + 15]
                # Replace Fortran 'D' with 'E'
                val_str = val_str.replace('D', 'E').replace('d', 'e')
                if not val_str.strip(): continue
                try:
                    raw_coeffs.append(float(val_str))
                except ValueError:
                    pass

        # Expecting at least 14 coefficients
        if len(raw_coeffs) >= 14:
            # High Temp Interval
            current_species.add_interval(t_mid, t_high, raw_coeffs[0:7])
            # Low Temp Interval
            current_species.add_interval(t_low, t_mid, raw_coeffs[7:14])

        self.species[name] = current_species
        return idx + 4  # Move past the 4 lines

    def _parse_species_nasa9(self, lines: List[str], idx: int) -> int:
        """
        Parses a single species block in NASA-9 format.
        """
        # Line 1: Header (Name, Comments) -> already read name before calling this,
        # but we need to ensure we are consistent.
        # Actually, in the main loop, we read Line 1 to get the name.
        # But wait, looking at your file structure:
        # H2 ... (Line 1)
        #  3 tpis78 H 2.00 ... (Line 2)

        # The 'name' was extracted in the main loop from Line 1.
        # So 'idx' passed here should point to Line 2?
        # Let's check the main loop logic.
        # Main loop: reads line (Line 1), detects format. If NASA-9, calls this with idx pointing to Line 1?
        # No, if it's NASA-9, the main loop sees the name on Line 1.
        # We need to process Line 2 to get Composition and MW.

        # Let's assume idx points to Line 1 (Name line).
        name = lines[idx].split()[0].strip()

        # Move to Line 2 (Composition line)
        idx += 1
        if idx >= len(lines): return idx
        line2 = lines[idx]

        # Parse Composition and MW from Line 2
        # --- FIX: Handle sticky columns in composition (e.g., "2.00O") ---
        # Insert a space between any digit and an uppercase letter
        # Example: "2.00O" -> "2.00 O"
        line2_clean = re.sub(r'(\d)([A-Z])', r'\1 \2', line2)
        # Example: " 3 tpis78 H   2.00    0.00    0.00    0.00    0.00 0    2.0158800          0.000"
        tokens = line2_clean.replace('D', 'E').split()

        composition = {}
        molecular_weight = 0.0

        # Simple parser for the tokens
        # Structure: [Intervals] [Code] [El1] [Num1] [El2] [Num2] ... [Charge] [MW] [Heat]
        # Heuristic: Scan for Element-Number pairs

        # We start skipping the first 2 tokens (Intervals, Code)
        # We iterate and look for alpha-strings as Elements.

        k = 0
        while k < len(tokens):
            token = tokens[k]

            # Look for Element-Number pairs
            # Valid element: Is alpha, len <= 2, and NOT strictly lowercase (like 'g' code)
            # Usually elements are 'H', 'N', 'Ar', 'Fe' (Title case or Upper)
            # Checking 'token[0].isupper()' helps filter out 'g', 'tpis' etc.
            if token.isalpha() and len(token) <= 2 and token[0].isupper():
                if k + 1 < len(tokens):
                    try:
                        amount = float(tokens[k + 1])
                        if amount > 0:
                            composition[token] = amount
                        k += 1  # Consume the number
                    except ValueError:
                        pass
            k += 1

        # Fallback for MW: Look for the specific pattern near end of line
        # Logic: MW is usually the 2nd to last float number on the line.
        # Let's collect all floats from the line
        floats_on_line = []
        for t in tokens:
            try:
                floats_on_line.append(float(t))
            except ValueError:
                pass

        # Usually standard line has: ... [Charge] [MW] [HeatOfFormation]
        # So MW is floats_on_line[-2]
        if len(floats_on_line) >= 2:
            molecular_weight = floats_on_line[-2]

        current_species = Species(name, composition, molecular_weight)

        # Line 3+: Intervals and Coefficients
        idx += 1

        while idx < len(lines):
            line = lines[idx]

            # Check for Interval Header: 2 floats + 1 int (usually)
            # "   200.000   1000.0007 -2.0 ..."
            parts = line.replace('D', 'E').split()

            # Check if this line is an Interval Header
            is_header = False
            if len(parts) >= 3:
                try:
                    t_min = float(parts[0])
                    t_max = float(parts[1])
                    # usually 3rd part is number of coeffs (7 or 9) or order
                    # In your file it is '7', followed by exponents.
                    is_header = True
                except ValueError:
                    is_header = False

            if is_header:
                t_min = float(parts[0])
                t_max = float(parts[1])

                # The coeffs follow in the next lines.
                # Since we identified it's 9-coeff format from the header (exponents present),
                # we need to read exactly 9 coefficients.
                # In your file, they seem to be on the NEXT lines (not same line).
                # Example:
                # Header: ...
                # Coeffs1: 5 nums
                # Coeffs2: 4 nums

                idx += 1
                coeffs = []
                while len(coeffs) < 9 and idx < len(lines):
                    cline = lines[idx]
                    # NASA-9 coeff lines are strictly formatted: 5 numbers of width 16.
                    # e.g. " 1.234D+02-5.678D-01..."

                    # Instead of split(), we strictly slice 16-char chunks.
                    # Line length is typically up to 80 chars.
                    row_vals = []
                    for pos in range(0, min(80, len(cline)), 16):
                        chunk = cline[pos:pos + 16]
                        # Must ensure chunk is not just whitespace or newline
                        if len(chunk.strip()) < 3:
                            continue

                        try:
                            val_str = chunk.replace('D', 'E').replace('d', 'e')
                            row_vals.append(float(val_str))
                        except ValueError:
                            pass

                    coeffs.extend(row_vals)
                    idx += 1

                if len(coeffs) >= 9:
                    current_species.add_interval(t_min, t_max, coeffs[:9])
                else:
                    # Something went wrong reading coeffs
                    break
            else:
                # Not a header, and not a coeff line (we are in outer loop).
                # Likely start of next species or END.
                break

        self.species[name] = current_species
        return idx

    def get_species(self, name: str) -> Optional[Species]:
        return self.species.get(name)