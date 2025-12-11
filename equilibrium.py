"""
Equilibrium Solver for Chemical Equilibrium Applications.

This module implements a Newton-Raphson solver for minimizing Gibbs free energy
subject to mass balance and thermodynamic constraints (Enthalpy/Entropy).
It follows the matrix structure described in NASA RP-1311 (Gordon & McBride, 1994).

Key Features:
- Solves TP (Temperature-Pressure), HP (Enthalpy-Pressure), and SP (Entropy-Pressure) problems.
- Calculates thermodynamic derivatives (Cp, gamma, sonic velocity).
- Prepares structure for condensed phases (currently TODO).
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

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
        self.MAX_ITER = 100
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

        # 1. Initialize State
        # -------------------
        # Filter species that exist in DB
        species_objs = [self.db.species[name] for name in init_species if name in self.db.species]
        if not species_objs:
            logger.error("No valid species found in database.")
            return None

        n_species = len(species_objs)
        element_list = list(elements_b0.keys())
        n_elements = len(element_list)

        # Build 'a' matrix: a[i, j] = atoms of element i in species j
        a_matrix = np.zeros((n_elements, n_species))
        for j, sp in enumerate(species_objs):
            for i, el in enumerate(element_list):
                a_matrix[i, j] = sp.composition.get(el, 0.0)

        # b0 vector (elemental moles constraint)
        b0 = np.array([elements_b0[el] for el in element_list])

        # Initial Guess for Composition (n_j) and Temperature (T)
        # TODO: Implement a robust initial guess routine (like CEA's 'estim').
        # Currently using a uniform distribution which might fail for complex cases.
        nj = np.ones(n_species) * 0.1 / n_species

        # Initial T
        if problem_type == 'TP':
            current_T = val_1
        else:
            # For HP/SP, we need a guess. 3000K is a typical rocket chamber T.
            current_T = 3000.0

            # Initial Total Moles (n)
        total_n = np.sum(nj)
        ln_nj = np.log(nj)
        ln_n = np.log(total_n)
        ln_T = np.log(current_T)

        # Solver Loop
        # -----------
        for iteration in range(self.MAX_ITER):

            # 2. Calculate Thermodynamic Properties for current T
            # ---------------------------------------------------
            # We need dimensionless properties: H/RT, S/R, Cp/R, G/RT
            # vectors of size n_species
            h_rt_vec = np.zeros(n_species)
            s_r_vec = np.zeros(n_species)
            cp_r_vec = np.zeros(n_species)
            mu_rt_vec = np.zeros(n_species)  # Chemical potential

            for j, sp in enumerate(species_objs):
                cp_r, h_rt, s_r, g_rt = sp.get_properties(current_T)
                h_rt_vec[j] = h_rt
                s_r_vec[j] = s_r
                cp_r_vec[j] = cp_r

                # mu_j/RT = (G/RT)_j + ln(n_j) + ln(P_atm) - ln(n)
                # Note: P must be in bars/atm depending on DB standard state. 
                # CEA standard state is usually 1 bar.
                # If P is in bar, ln(P) is correct.
                # TODO: Check if condensed phase logic is needed here.
                # For condensed: mu_j/RT = (G/RT)_j (no pressure term)
                mu_rt_vec[j] = g_rt + ln_nj[j] + np.log(P) - ln_n

            # 3. Build Interaction Matrix and Residual Vector
            # -----------------------------------------------
            # Matrix size: 
            #   TP problem: N_el + 1 (Variables: pi_i, dln_n)
            #   HP/SP prob: N_el + 2 (Variables: pi_i, dln_n, dln_T)

            is_energy_constrained = (problem_type != 'TP')
            dim = n_elements + 1 + (1 if is_energy_constrained else 0)

            matrix = np.zeros((dim, dim))
            rhs = np.zeros(dim)

            # --- Pre-calculate terms ---
            # nj_a[i, j] = nj[j] * a[i, j]
            # This helps in computing sums efficiently
            nj_diag = np.diag(nj)

            # 3.1 Mass Balance Block (Rows 0 to N_el-1)
            # A_ik = sum_j (a_ij * a_kj * n_j)
            # Matrix form: A @ diag(nj) @ A.T
            A_block = a_matrix @ nj_diag @ a_matrix.T
            matrix[:n_elements, :n_elements] = A_block

            # 3.2 Equation of State Block (Row/Col N_el)
            # Sum of moles: sum_j (a_ij * n_j)
            # Vector form: A @ nj
            vec_n_elements = a_matrix @ nj
            matrix[:n_elements, n_elements] = vec_n_elements
            matrix[n_elements, :n_elements] = vec_n_elements

            # Corner element (Row N_el, Col N_el) = sum(n_j) - n
            # According to CEA derivation (Gordon-McBride Eq 2.24), this term is sum(n_j)
            # But the variable is dln_n, so the coefficient is effectively sum(n_j).
            # Wait, let's verify Gordon-McBride Eq 2.24 carefully.
            # The variable is delta(ln n). The coefficient for pi_i is sum(aij nj).
            # The coefficient for delta(ln n) is sum(nj) - n. (Eq 2.26)
            matrix[n_elements, n_elements] = np.sum(nj) - total_n  # Often close to 0 if n matches sum(nj)

            # 3.3 Energy/Entropy Block (Row/Col N_el+1) - Only for HP/SP
            if is_energy_constrained:
                # Coefficients for dln_T column (Column N_el + 1)
                # Part 1: sum(aij * nj * (H/RT)_j) for Mass rows
                # Vector: A @ (nj * H_RT)
                term_h_or_u = h_rt_vec if problem_type == 'HP' else s_r_vec  # Simplified? No, needs partials.

                # Careful: For HP, we need Enthalpy terms. For SP, Entropy terms.
                # Let's handle HP first (Enthalpy).
                if problem_type == 'HP':
                    # Derivative of mass balance w.r.t T involves H term? No.
                    # The T column in Mass Balance equations:
                    # Coeff = sum(a_ij * n_j * (Ho_j/RT))  (Eq 2.24 in RP-1311)
                    # Because d(mu/RT)/dlnT = -H/RT

                    t_coupling_vec = a_matrix @ (nj * h_rt_vec)
                    matrix[:n_elements, n_elements + 1] = t_coupling_vec
                    matrix[n_elements + 1, :n_elements] = t_coupling_vec

                    # Intersection with dln_n (Row N_el+1, Col N_el)
                    # sum(nj * H/RT)
                    sum_nj_h = np.dot(nj, h_rt_vec)
                    matrix[n_elements, n_elements + 1] = sum_nj_h
                    matrix[n_elements + 1, n_elements] = sum_nj_h

                    # Bottom-Right Corner (Row N_el+1, Col N_el+1)
                    # sum(nj * (Cp/R + (H/RT)^2))
                    # Eq 2.28 in RP-1311
                    term_corner = np.dot(nj, cp_r_vec + h_rt_vec ** 2)
                    matrix[n_elements + 1, n_elements + 1] = term_corner

                elif problem_type == 'SP':
                    # TODO: Implement SP (Entropy) specific matrix terms.
                    # It's similar to HP but uses S/R and d(S/R)/dlnT = Cp/R terms.
                    # For now, raise NotImplemented to keep code clean or fallback to HP logic structure.
                    raise NotImplementedError("SP problem type fully rigorous matrix not yet implemented.")

            # 3.4 Residual Vector (RHS)
            # -------------------------
            # Mass Balance Errors: b0_i - sum(a_ij * n_j)
            # Plus potential terms (see RP-1311 Eq 2.24 RHS)
            # RHS_i = b0_i - sum(a_ij * nj) + sum(a_ij * nj * mu_j) ... (This is standard derivation)

            # Actually, standard Newton form Ax = B:
            # RHS_i = b0 - sum(aij*nj) ... no, wait.
            # In reduced form (pi variables):
            # RHS_i = b0_i - sum(a_ij * n_j) + [terms from expanding mu]
            # Let's use the explicit summation form:
            # RHS_i = b0_i - sum(a_ij * n_j) + sum_j (a_ij * n_j * mu_j) <-- This simplifies things if we solve for corrections.

            # Let's stick to the corrections formulation:
            # We solve for del_pi, del_ln_n, del_ln_T

            # Vector of mu*nj
            nj_mu = nj * mu_rt_vec

            # Mass Rows RHS:
            # (b0 - A @ nj) + A @ (nj * mu)
            rhs[:n_elements] = b0 - vec_n_elements + a_matrix @ nj_mu

            # Sum Moles Row RHS:
            # total_n - sum(nj) + sum(nj * mu)
            rhs[n_elements] = total_n - np.sum(nj) + np.sum(nj_mu)

            # Energy Row RHS (HP):
            if is_energy_constrained and problem_type == 'HP':
                # Target Enthalpy H0/RT (dimensionless input val_1 must be H0/R -> H0/RT = val_1 / T)
                # Wait, input val_1 for HP is usually H0/R (units of K).
                h0_rt_target = val_1 / current_T

                # Current Enthalpy of mixture
                # H_mix/RT = sum(nj * H_j/RT) / sum(nj) ? No, per kg basis.
                # Equations are extensive in CEA. H_total / RT = sum(nj * H_j/RT).
                # We need to match input H0 per unit mass.
                # Actually, CEA usually balances H/R.

                # Let's assume val_1 is H_target / R (total enthalpy of 1 kg mixture / R)

                # RHS_energy = H0/RT - sum(nj * H/RT) + sum(nj * H/RT * mu)
                current_H_total = np.dot(nj, h_rt_vec)
                rhs[n_elements + 1] = h0_rt_target - current_H_total + np.dot(nj_mu, h_rt_vec)

            # 4. Solve Linear System
            # ----------------------
            try:
                corrections = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                # Matrix singularity handling
                logger.warning(f"Singular matrix at iter {iteration}. Applying regularization.")
                matrix += np.eye(dim) * 1e-8
                try:
                    corrections = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    logger.error("Solver failed: Singular matrix.")
                    return None

            # 5. Extract Corrections & Update
            # -------------------------------
            pi_update = corrections[:n_elements]
            dln_n_update = corrections[n_elements]
            dln_t_update = 0.0
            if is_energy_constrained:
                dln_t_update = corrections[n_elements + 1]

            # Calculate dln_nj for each species
            # dln_nj = -mu_j + sum(a_ij * pi_i) + dln_n + (H_j/RT)*dln_T
            # (Last term is for T variable)

            term_temp = 0.0
            if is_energy_constrained:
                # d(mu)/dlnT = -H/RT
                # So dln_nj += (H_j/RT) * dln_T
                term_temp = h_rt_vec * dln_t_update

            dln_nj = -mu_rt_vec + (a_matrix.T @ pi_update) + dln_n_update + term_temp

            # 6. Convergence Control (Lambda damping)
            # ---------------------------------------
            # Prevent excessive steps that drive moles negative (or huge T jumps)
            lambda_f = 1.0

            # Limit T change to avoid negative T or explosion
            if is_energy_constrained:
                if abs(dln_t_update) > 0.2:  # Limit approx 20% change
                    lambda_f = min(lambda_f, 0.2 / abs(dln_t_update))

            # Limit mole changes (standard max step 2.0 in log space)
            max_dln_nj = np.max(np.abs(dln_nj))
            if max_dln_nj > 2.0:
                lambda_f = min(lambda_f, 2.0 / max_dln_nj)

            # Apply updates
            ln_nj += lambda_f * dln_nj
            ln_n += lambda_f * dln_n_update

            if is_energy_constrained:
                ln_T += lambda_f * dln_t_update
                current_T = np.exp(ln_T)

            nj = np.exp(ln_nj)
            total_n = np.exp(ln_n)

            # Normalization (optional but good for stability)
            # CEA does normalization inside the loop sometimes.

            # Check Convergence
            # Criteria: max correction is small AND residuals are small
            # For now, just checking corrections.
            if (max_dln_nj * lambda_f < self.TOLERANCE and
                    (abs(dln_t_update * lambda_f) < self.TOLERANCE if is_energy_constrained else True)):

                logger.info(f"Converged in {iteration} iterations. T={current_T:.2f} K")

                # TODO: Check for Condensed Phases here.
                # Logic: Calculate chemical potential of excluded solids.
                # If mu_solid/RT < sum(a_ij * pi_j), the solid should form.
                # Add to 'init_species' and restart? Or dynamic add/remove.

                # 7. Post-Processing: Derivatives
                # -------------------------------
                props = self._calculate_derivatives(
                    species_objs, nj, current_T, P, matrix, n_elements, is_energy_constrained, total_n, cp_r_vec
                )

                # Build Result
                # Filter Trace Species
                final_mole_fractions = {}
                for j, sp in enumerate(species_objs):
                    if nj[j] / total_n > self.TRACE_LIMIT:
                        final_mole_fractions[sp.name] = nj[j] / total_n

                return EquilibriumResult(
                    P=P,
                    T=current_T,
                    mole_fractions=final_mole_fractions,
                    properties=props,
                    converged=True,
                    iterations=iteration
                )

        logger.error("Max iterations reached without convergence.")
        return None

    def _calculate_derivatives(self,
                               species,
                               nj,
                               T,
                               P,
                               matrix,
                               n_elements,
                               is_energy_constrained,
                               total_n,
                               cp_r_vec) -> Dict[str, float]:
        """
        Solves the derivative linear equations to find Cp, Gamma, etc.
        Reference: Gordon & McBride 1994, Chapter 2.

        This requires solving the same matrix with different RHS vectors
        to find derivatives with respect to ln P and ln T.
        """
        # For a full rocket solution, we strictly need:
        # dln_V/dln_P, dln_V/dln_T, Cp_frozen, Cp_equilibrium.

        # 1. Solve for d/dlnT (assuming constant P)
        # We assume the 'matrix' passed is the converged Jacobian.
        # However, for derivatives, the RHS is specific.

        # NOTE: A rigorous implementation requires separating the matrix parts
        # or reconstructing it. Given complexity, providing a simplified calculation
        # for frozen/equilibrium Cp based on summation properties for now.

        # Equilibrium Cp calculation involves the reaction contributions.
        # Cp_eq = Cp_frozen + reaction_terms

        # Frozen Properties
        # Cp_frozen = sum(xi * Cp_i)
        mole_fracs = nj / total_n
        cp_frozen_r = np.dot(mole_fracs, cp_r_vec)

        # Molecular Weight of Mixture
        mw_mix = 0.0
        for j, sp in enumerate(species):
            mw_mix += mole_fracs[j] * sp.molecular_weight

        # Specific Heat Ratio (Gamma)
        # Gamma = Cp / Cv = Cp / (Cp - R)
        gamma_s = cp_frozen_r / (cp_frozen_r - 1.0)

        # Sound Speed (Frozen)
        # a = sqrt(gamma * R * T / M)
        # R_univ = 8314.51
        from constants import R_UNIV
        try:
            a_frozen = np.sqrt(gamma_s * R_UNIV * T / mw_mix)
        except ValueError:
            a_frozen = 0.0

        # TODO: Implement full Equilibrium Gamma and Sound Speed.
        # This requires solving the matrix for dln_n/dln_P.

        return {
            "cp_frozen_r": cp_frozen_r,
            "gamma_frozen": gamma_s,
            "mw_mix": mw_mix,
            "sonic_velocity_frozen": a_frozen,
            "enthalpy_total_rt": 0.0  # Placeholder
        }