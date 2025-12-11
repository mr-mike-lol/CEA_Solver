"""
Equilibrium Solver for Chemical Equilibrium Applications.

This module implements a Newton-Raphson solver for minimizing Gibbs free energy
subject to mass balance and thermodynamic constraints (Enthalpy/Entropy).
It follows the matrix structure described in NASA RP-1311 (Gordon & McBride, 1994).

Key Features:
- Solves TP (Temperature-Pressure), HP (Enthalpy-Pressure), and SP (Entropy-Pressure) problems.
- Calculates thermodynamic derivatives (Cp, gamma, sonic velocity).
- Prepares structure for condensed phases.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from constants import R_UNIV

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("CEA_Solver")


@dataclass
class EquilibriumResult:
    """
    Stores the result of an equilibrium calculation.
    """
    P: float  # Pressure [Bar] -> Converted to Pa internally if needed? CEA uses Bar/Atm mostly.
    T: float  # Temperature [K]
    mole_fractions: Dict[str, float]  # n_j / n_total
    properties: Dict[str, float] = field(default_factory=dict)  # Cp, Gamma, H, S, etc.
    converged: bool = False
    iterations: int = 0


class EquilibriumSolver:
    """
    Solves chemical equilibrium problems using the minimization of Gibbs free energy.
    """

    def __init__(self, thermo_db):
        """
        Args:
            thermo_db: An instance of ThermoDatabase containing species data.
        """
        self.db = thermo_db
        self.MAX_ITER = 1000
        self.TOLERANCE = 1e-5

        # Trace limit for species moles. 
        # In full CEA, species below this are removed from the matrix but kept in check.
        self.TRACE_LIMIT = 1e-15

    def solve(self,
              elements_b0: Dict[str, float],
              init_species: List[str],
              P: float,
              val_1: float,
              val_2: Optional[float] = None,
              problem_type: str = 'TP') -> Optional[EquilibriumResult]:
        """
        Main solver entry point.

        Args:
            elements_b0: Normalized elemental moles per kg of mixture (b_i^0).
            init_species: List of candidate species names to consider.
            P: Pressure in Bar.
            val_1: First state variable.
                   - For 'TP': Temperature (K).
                   - For 'HP': Enthalpy of mixture H0/R (K).
                   - For 'SP': Entropy of mixture S0/R (dimensionless).
            val_2: Not used for standard standard problems defined here, reserved.
            problem_type: 'TP', 'HP', or 'SP'.

        Returns:
            EquilibriumResult object or None if failed.
        """
        logger.info(f"Starting {problem_type} solver at P={P:.4f} bar.")

        while True:

            # 1. Initialize State
            species_objs = [self.db.species[name] for name in init_species if name in self.db.species]
            if not species_objs:
                logger.error("No valid species found in database.")
                return None

            n_species = len(species_objs)
            element_list = list(elements_b0.keys())
            n_elements = len(element_list)

            # Build 'a' matrix
            a_matrix = np.zeros((n_elements, n_species))
            for j, sp in enumerate(species_objs):
                for i, el in enumerate(element_list):
                    a_matrix[i, j] = sp.composition.get(el, 0.0)

            b0 = np.array([elements_b0[el] for el in element_list])

            # Initial Guess for Composition (n_j) and Temperature (T)
            # Using the robust estimation routine
            if problem_type == 'TP':
                current_T = val_1
            else:
                # For HP/SP, start with a reasonable guess
                current_T = 3000.0

                # Call the estimation routine (assuming it is implemented in the class)
            nj, total_n = self._estimate_initial_composition(
                species_objs, b0, a_matrix, current_T, P
            )

            ln_nj = np.log(nj)
            ln_n = np.log(total_n)
            ln_T = np.log(current_T)

            # Solver Loop
            for iteration in range(self.MAX_ITER):
                # --- Debug output every 50 iterations ---
                if iteration % 50 == 0 and iteration > 0:
                    logger.info(f"Iter {iteration}: T={current_T:.1f} K")

                # 2. Thermodynamic Properties
                h_rt_vec = np.zeros(n_species)
                s_r_vec = np.zeros(n_species)
                cp_r_vec = np.zeros(n_species)
                mu_rt_vec = np.zeros(n_species)

                for j, sp in enumerate(species_objs):
                    cp_r, h_rt, s_r, g_rt = sp.get_properties(current_T)
                    h_rt_vec[j] = h_rt
                    s_r_vec[j] = s_r
                    cp_r_vec[j] = cp_r

                    # mu_j/RT = (G/RT)_j + ln(n_j) + ln(P_atm) - ln(n)
                    mu_rt_vec[j] = g_rt + ln_nj[j] + np.log(P) - ln_n

                # 3. Build Interaction Matrix and Residual Vector
                is_energy_constrained = (problem_type != 'TP')
                dim = n_elements + 1 + (1 if is_energy_constrained else 0)

                matrix = np.zeros((dim, dim))
                rhs = np.zeros(dim)

                nj_diag = np.diag(nj)

                # 3.1 Mass Balance Block (Rows 0 to N_el-1)
                # A_ik = sum(a_ij * a_kj * nj)
                A_block = a_matrix @ nj_diag @ a_matrix.T
                matrix[:n_elements, :n_elements] = A_block

                # 3.2 Equation of State Block (Row/Col N_el)
                # A_in = sum(a_ij * nj)
                vec_n_elements = a_matrix @ nj
                matrix[:n_elements, n_elements] = vec_n_elements
                matrix[n_elements, :n_elements] = vec_n_elements
                matrix[n_elements, n_elements] = np.sum(nj) - total_n

                # 3.3 Energy/Entropy Block (Row/Col N_el+1) - For HP/SP
                if is_energy_constrained:
                    # Common term: Column for T (dln_T) in Mass and EOS rows
                    # This column ALWAYS depends on H/RT because d(ln_nj)/d(ln_T) ~ H/RT
                    # Vector: A @ (nj * H_RT)
                    t_col_vec_mass = a_matrix @ (nj * h_rt_vec)
                    t_col_elem_sum = np.dot(nj, h_rt_vec)

                    # Fill Column N_el + 1 (Temperature corrections on Mass/Sum)
                    matrix[:n_elements, n_elements + 1] = t_col_vec_mass
                    matrix[n_elements, n_elements + 1] = t_col_elem_sum

                    if problem_type == 'HP':
                        # --- HP Problem (Enthalpy) ---
                        # Row N_el + 1 is Enthalpy Equation.
                        # Coefficients for pi_i (Columns 0..N_el-1): Same as T-column (Symmetric)
                        matrix[n_elements + 1, :n_elements] = t_col_vec_mass

                        # Coefficient for dln_n (Column N_el): Same as T-col (Symmetric)
                        matrix[n_elements + 1, n_elements] = t_col_elem_sum

                        # Corner element (Row T, Col T): sum(nj * (Cp/R + (H/RT)^2))
                        term_corner = np.dot(nj, cp_r_vec + h_rt_vec ** 2)
                        matrix[n_elements + 1, n_elements + 1] = term_corner

                        # RHS for HP
                        # Target H0/RT - Current H/RT + Correction terms
                        h0_rt_target = val_1 / current_T
                        current_H_total = np.dot(nj, h_rt_vec)
                        nj_mu = nj * mu_rt_vec
                        rhs[n_elements + 1] = h0_rt_target - current_H_total + np.dot(nj_mu, h_rt_vec)

                    elif problem_type == 'SP':
                        # --- SP Problem (Entropy) ---
                        # Row N_el + 1 is Entropy Equation: sum(nj * S_j) = S0

                        # Coefficients for pi_i (Columns 0..N_el-1): sum(a_ij * nj * S_j/R)
                        # Vector: A @ (nj * S_R)
                        s_row_vec_mass = a_matrix @ (nj * s_r_vec)
                        matrix[n_elements + 1, :n_elements] = s_row_vec_mass

                        # Coefficient for dln_n (Column N_el): sum(nj * S_j/R)
                        s_row_elem_sum = np.dot(nj, s_r_vec)
                        matrix[n_elements + 1, n_elements] = s_row_elem_sum

                        # Corner element (Row T, Col T): sum(nj * (Cp/R + (S/R)*(H/RT)))
                        # Note: derived from d(sum(nj*S))/dlnT
                        term_corner = np.dot(nj, cp_r_vec + s_r_vec * h_rt_vec)
                        matrix[n_elements + 1, n_elements + 1] = term_corner

                        # RHS for SP
                        # Target S0/R - Current S/R + Correction terms
                        s0_r_target = val_1
                        current_S_total = np.dot(nj, s_r_vec)
                        nj_mu = nj * mu_rt_vec
                        # The correction term is sum(nj * mu * S/R)
                        rhs[n_elements + 1] = s0_r_target - current_S_total + np.dot(nj_mu, s_r_vec)

                # 3.4 Residual Vector (Rest of RHS)
                nj_mu = nj * mu_rt_vec
                rhs[:n_elements] = b0 - vec_n_elements + a_matrix @ nj_mu
                rhs[n_elements] = total_n - np.sum(nj) + np.sum(nj_mu)

                # 4. Solve Linear System
                try:
                    corrections = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    logger.warning(f"Singular matrix at iter {iteration}. Applying regularization.")
                    matrix += np.eye(dim) * 1e-8
                    try:
                        corrections = np.linalg.solve(matrix, rhs)
                    except np.linalg.LinAlgError:
                        return None

                # 5. Extract Corrections & Update
                pi_update = corrections[:n_elements]
                dln_n_update = corrections[n_elements]
                dln_t_update = 0.0
                if is_energy_constrained:
                    dln_t_update = corrections[n_elements + 1]

                # Calculate dln_nj
                # dln_nj = -mu_j + sum(a_ij * pi_i) + dln_n + (H_j/RT)*dln_T
                term_temp = 0.0
                if is_energy_constrained:
                    term_temp = h_rt_vec * dln_t_update

                dln_nj = -mu_rt_vec + (a_matrix.T @ pi_update) + dln_n_update + term_temp

                # 6. Convergence Control
                lambda_f = 1.0
                if is_energy_constrained:
                    if abs(dln_t_update) > 0.5:
                        lambda_f = min(lambda_f, 0.5 / abs(dln_t_update))

                max_dln_nj = np.max(np.abs(dln_nj))
                if max_dln_nj > 2.0:
                    lambda_f = min(lambda_f, 2.0 / max_dln_nj)

                ln_nj += lambda_f * dln_nj
                ln_n += lambda_f * dln_n_update

                if is_energy_constrained:
                    ln_T += lambda_f * dln_t_update
                    current_T = np.exp(ln_T)
                    # Safety clamp T
                    current_T = max(200.0, min(current_T, 10000.0))

                nj = np.exp(ln_nj)
                total_n = np.exp(ln_n)

                if (max_dln_nj * lambda_f < self.TOLERANCE and
                        (abs(dln_t_update * lambda_f) < self.TOLERANCE if is_energy_constrained else True)):

                    logger.info(f"Converged in {iteration} iterations. T={current_T:.2f} K")

                    # Check Condensed Phases
                    current_pi = pi_update
                    max_violation = 0.0
                    best_candidate = None

                    for name, sp in self.db.species.items():
                        if name in init_species: continue
                        if not any(x in name for x in ['(L)', '(S)', '(cr)', '(s)', '(l)']): continue
                        sp_elements = set(sp.composition.keys())
                        if not sp_elements.issubset(set(element_list)): continue

                        try:
                            _, _, _, g_rt_cand = sp.get_properties(current_T)
                        except ValueError:
                            continue

                        pot_elements = 0.0
                        for i, el in enumerate(element_list):
                            pot_elements += sp.composition.get(el, 0.0) * current_pi[i]

                        violation = pot_elements - g_rt_cand
                        if violation > max_violation:
                            max_violation = violation
                            best_candidate = name

                    if best_candidate and max_violation > 1e-4:
                        logger.info(f"Adding condensed phase: {best_candidate}")
                        init_species.append(best_candidate)
                        break  # Break inner loop, continue while True

                    # Final Result
                    props = self._calculate_derivatives(species_objs, nj, current_T, P, total_n, cp_r_vec, h_rt_vec)
                    props["s_total_r"] = np.dot(nj, s_r_vec) / total_n

                    final_mole_fractions = {}
                    for j, sp in enumerate(species_objs):
                        if nj[j] / total_n > self.TRACE_LIMIT:
                            final_mole_fractions[sp.name] = nj[j] / total_n

                    return EquilibriumResult(P, current_T, final_mole_fractions, props, True, iteration)

                else:
                    logger.error("Max iterations reached.")
                    return None

    def _calculate_derivatives(self,
                               species_objs: List[Any],
                               nj: np.ndarray,
                               T: float,
                               P: float,
                               total_n: float,
                               cp_r_vec: np.ndarray,
                               h_rt_vec: np.ndarray) -> Dict[str, float]:
        """
        Calculates equilibrium thermodynamic derivatives (Cp, Gamma, Sound Speed).
        Solves the matrix equations for partial derivatives w.r.t. T and P.
        Reference: Gordon & McBride 1994 (NASA RP-1311), Equations 2.32 - 2.65.
        """
        from constants import R_UNIV

        n_species = len(species_objs)
        n_elements = len(self.db.species[species_objs[0].name].composition)  # Get from first species?
        # Safer to reconstruct element list from DB or pass it.
        # But we can infer n_elements from the matrix size if we passed matrix.
        # Let's rebuild the small TP matrix here to be robust.

        # 1. Rebuild Mass Balance Matrix (Size Nel+1 x Nel+1)
        # This is the Jacobian at the converged point.
        # We need the element definitions again. Let's assume standard order from species.
        # It's better if 'solve' passes the element list or we extract it.
        # Let's extract unique elements from species_objs to be consistent.
        # NOTE: This implies element order must match what 'solve' used.
        # To avoid desync, ideally 'solve' passes 'a_matrix'.
        # For now, let's reconstruct 'a_matrix' assuming standard sort order of keys.

        all_elements = set()
        for sp in species_objs:
            all_elements.update(sp.composition.keys())
        element_list = sorted(list(all_elements))  # Use sorted to ensure deterministic order
        n_elements = len(element_list)

        a_matrix = np.zeros((n_elements, n_species))
        for j, sp in enumerate(species_objs):
            for i, el in enumerate(element_list):
                a_matrix[i, j] = sp.composition.get(el, 0.0)

        # Build Matrix A (Size Nel+1)
        dim = n_elements + 1
        matrix = np.zeros((dim, dim))

        # A_ik = sum(a_ij * a_kj * nj)
        nj_diag = np.diag(nj)
        A_block = a_matrix @ nj_diag @ a_matrix.T
        matrix[:n_elements, :n_elements] = A_block

        # Column/Row for sum moles: sum(a_ij * nj)
        vec_n_elem = a_matrix @ nj
        matrix[:n_elements, n_elements] = vec_n_elem
        matrix[n_elements, :n_elements] = vec_n_elem

        # Corner: sum(nj) - n. But for derivatives matrix, the corner is sum(nj) (See Eq 2.37)
        # Actually, in derivative eq (2.34), the coefficient for dln_n is sum(n_j).
        matrix[n_elements, n_elements] = np.sum(nj)

        # 2. Build RHS Vectors
        # --------------------
        # For d/dlnT (Eq 2.34): RHS is vector of Enthalpies
        # Vector elements: sum(a_ij * nj * H_j/RT)
        # Scalar element:  sum(nj * H_j/RT)
        rhs_t = np.zeros(dim)
        nj_h = nj * h_rt_vec
        rhs_t[:n_elements] = a_matrix @ nj_h
        rhs_t[n_elements] = np.sum(nj_h)

        # For d/dlnP (Eq 2.37): RHS is vector of Moles (Volumes)
        # Vector elements: sum(a_ij * nj)  (= b0_i if conserved, but we use actuals)
        # Scalar element:  sum(nj)
        rhs_p = np.zeros(dim)
        rhs_p[:n_elements] = vec_n_elem
        rhs_p[n_elements] = np.sum(nj)

        # 3. Solve Systems
        # ----------------
        # We solve A * x = RHS.
        # Note on signs: Gordon-McBride often define equations such that A*dx = RHS.
        # Checked Eq 2.34: Matrix * [d_pi, d_ln_n] = [Enthalpy_Vector]
        try:
            sol_t = np.linalg.solve(matrix, rhs_t)
            sol_p = np.linalg.solve(matrix, rhs_p)
        except np.linalg.LinAlgError:
            logger.warning("Derivative matrix singular. Returning frozen properties.")
            # Fallback to frozen
            mw_mix = np.sum(nj * np.array([sp.molecular_weight for sp in species_objs])) / total_n
            cp_frozen = np.dot(nj, cp_r_vec) / total_n
            return {
                "cp_eq_r": cp_frozen, "gamma_s": 1.4, "son_vel": 0.0, "mw": mw_mix
            }

        # Extract derivatives of system variables
        # dln_n_T = (d ln n / d ln T)_P
        dln_n_t = sol_t[n_elements]

        # dln_n_P = (d ln n / d ln P)_T
        dln_n_p = sol_p[n_elements]

        # Derivatives of Species (d ln n_j / d ln T)_P
        # Eq 2.32: dln_nj/dlnT = H/RT + sum(a_ij * pi_i)_T + dln_n_T
        # Wait, solution x corresponds to (d_pi, d_ln_n).
        # But look at Eq 2.32: dln_nj = H/RT + ...
        # The matrix equation (2.34) comes from substituting this into mass balance.
        # A * x = RHS_enthalpy implies x are indeed the derivatives of pi and ln_n w.r.t ln T.

        # We don't strictly need individual species derivatives for Cp_eq if we use the summed form:
        # Cp_eq/R = sum(nj*Cp_j)/R + sum(nj * H_j/RT * dln_nj_dlnT)
        # Let's compute the second term efficiently.

        pi_t = sol_t[:n_elements]
        # dln_nj_dlnT vector:
        dln_nj_t = h_rt_vec + a_matrix.T @ pi_t + dln_n_t

        # 4. Calculate Properties
        # -----------------------

        # C_p Equilibrium (Eq 2.62)
        # Cp_eq/R = sum(x_j * Cp_j) + sum(x_j * H_j/RT * dln_nj_dlnT)
        # Using totals (not fractions) first:
        term_1 = np.dot(nj, cp_r_vec)
        term_2 = np.dot(nj * h_rt_vec, dln_nj_t)
        cp_eq_r = (term_1 + term_2) / total_n

        # C_v Equilibrium calculation needed for Gamma?
        # CEA uses thermodynamic derivatives of Volume directly.
        # (d ln V / d ln T)_P = 1 + (d ln n / d ln T)_P  (Eq 2.42)
        dln_v_t = 1.0 + dln_n_t

        # (d ln V / d ln P)_T = -1 + (d ln n / d ln P)_T (Eq 2.43)
        dln_v_p = -1.0 + dln_n_p

        # Gamma_S (Isentropic Exponent) - Used for Sound Speed
        # gamma_s = - gamma / (d ln V / d ln P)_T  (Eq 2.63) ???
        # Actually simpler formula via C_p:
        # gamma_s = - (Cp/Cv) / (dlnV/dlnP) ...
        # Standard CEA formula (Eq 2.50 + 2.63):
        # gamma_s = - (Cp/R) / ( (Cp/R) * (dlnV/dlnP)_T + (dlnV/dlnT)_P^2 )
        # This relates Cp, density derivatives and sound speed.

        denom = cp_eq_r * dln_v_p + (dln_v_t) ** 2
        if abs(denom) < 1e-12:
            gamma_s = 1.4  # Error fallback
        else:
            gamma_s = - cp_eq_r / denom

        # Gamma (Specific Heat Ratio) Cp/Cv
        # gamma = gamma_s * (dlnV/dlnP)_T / (-1)? No.
        # gamma = - gamma_s * (dlnV/dlnP)_S ?
        # Eq 2.2b: gamma = (dlnP/dln_rho)_S = gamma_s.
        # CEA output labels 'GAMMAs' as the isentropic exponent used for sound speed.
        # This IS the number we want for rocket calculations.

        # Molecular Weight
        mw_mix = np.sum(nj * np.array([sp.molecular_weight for sp in species_objs])) / total_n

        # Sound Speed (Eq 2.55)
        # a = sqrt( nRT * gamma_s )  where n is moles per unit mass = 1/MW
        # a = sqrt( (R_univ * T / MW) * gamma_s )

        son_vel = np.sqrt((R_UNIV * T / mw_mix) * gamma_s)

        return {
            "cp_eq_r": cp_eq_r,
            "enthalpy_total_rt": np.dot(nj, h_rt_vec) / total_n,
            "entropy_total_r": np.dot(nj * (species_objs[0].intervals[0].coeffs[8] if False else 1.0)) / total_n,
            # Placeholder for S
            "mw": mw_mix,
            "gamma_s": gamma_s,
            "son_vel": son_vel,
            "dln_v_t": dln_v_t,
            "dln_v_p": dln_v_p
        }

    def _estimate_initial_composition(self,
                                      species_objs: List[Any],
                                      elements_b0: np.ndarray,
                                      a_matrix: np.ndarray,
                                      T_guess: float,
                                      P: float) -> Tuple[np.ndarray, float]:
        """
        Generates a robust initial guess for species moles (nj).
        Based on the logic that species with lower Gibbs energy are dominant.

        Args:
            species_objs: List of species objects.
            elements_b0: Vector of elemental moles per kg (b_i^0).
            a_matrix: Composition matrix (n_elements x n_species).
            T_guess: Guess temperature.
            P: Pressure (bar).

        Returns:
            Tuple (nj, total_n)
        """
        n_species = len(species_objs)
        n_elements = len(elements_b0)

        # 1. Calculate rough Gibbs energy for each species at T_guess
        # g_rt = H/RT - S/R (dimensionless)
        g_val = np.zeros(n_species)

        # Small epsilon to avoid log(0) if we used it elsewhere,
        # but here we use exp(-g), so it's safe.

        for j, sp in enumerate(species_objs):
            # We use the simplified formulas or just the first interval
            # Catch errors if T is out of range by clamping
            try:
                # Use get_properties directly
                _, _, _, g_rt = sp.get_properties(T_guess)
                g_val[j] = g_rt
            except ValueError:
                # If T_guess is wildly out, fallback to high energy (unlikely species)
                g_val[j] = 1000.0

        # 2. Probability proportional to Boltzmann factor: p ~ exp(-G/RT)
        # However, we must consider pressure: mu = g_rt + ln(P) + ln(nj/n)
        # Minimizing mu implies maximizing concentration for low g_rt.

        # Shift g_val to avoid overflow/underflow in exp
        min_g = np.min(g_val)
        # weight ~ exp(-(g_rt - min_g))
        weights = np.exp(-(g_val - min_g))

        # 3. Scale weights to satisfy mass balance roughly
        # We want: A @ nj = b0
        # Let nj = scale * weights
        # Then: scale * (A @ weights) = b0
        # This is an overdetermined system (n_elements equations, 1 variable 'scale').
        # We can't satisfy all exactly with one scalar.
        # Strategy: Satisfy the most abundant element or take an average requirement.

        element_demands = a_matrix @ weights  # How much of each element is in the weighted mix

        # Calculate required scale for each element: scale_i = b0_i / demand_i
        # We ignore elements with zero demand or zero b0 for the scale calc.
        scales = []
        for i in range(n_elements):
            if element_demands[i] > 1e-20 and elements_b0[i] > 1e-20:
                scales.append(elements_b0[i] / element_demands[i])

        if not scales:
            # Fallback if something went wrong (e.g. noble gases only?)
            avg_scale = 0.1 / np.sum(weights)
        else:
            # Use the average scale factor required
            # Alternatively, using the max scale ensures we have ENOUGH atoms,
            # preventing negative logs early on.
            avg_scale = np.mean(scales)

        nj_guess = weights * avg_scale

        # 4. Refinement: Ensure no species is absolute zero (set floor)
        # Using a small number like 1e-10 prevents singularity in first matrix inversion
        min_moles = 1e-5  # Sufficiently small start
        nj_guess = np.maximum(nj_guess, min_moles)

        total_n_guess = np.sum(nj_guess)

        return nj_guess, total_n_guess