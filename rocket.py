"""
Rocket Engine Performance Calculator.

This module uses the EquilibriumSolver to calculate theoretical rocket performance:
1. Combustion Chamber (HP problem)
2. Throat conditions (Isentropic expansion to Mach 1)
3. Nozzle Exit conditions (Isentropic expansion to P_exit or Area Ratio)

References:
    NASA RP-1311 (CEA descriptions)
    Sutton & Biblarz, Rocket Propulsion Elements
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple

# Import our custom modules
from equilibrium import EquilibriumSolver, EquilibriumResult
from thermo import ThermoDatabase
from constants import R_UNIV


class PropellantInput:
    """
    Helper class to prepare input for the solver (b0 and Enthalpy).
    Assumes reactants are ideal gases or phases available in standard thermo.inp.
    """

    def __init__(self, db: ThermoDatabase):
        self.db = db

    def calculate_inputs(self,
                         oxidizer: str,
                         fuel: str,
                         of_ratio: float,
                         ox_temp: float,
                         fuel_temp: float) -> Tuple[Dict[str, float], float]:
        """
        Calculates elemental moles per kg (b0) and mixture enthalpy (H0/R).

        Args:
            oxidizer: Name of oxidizer species in DB (e.g., "O2").
            fuel: Name of fuel species in DB (e.g., "H2").
            of_ratio: Oxidizer-to-Fuel mass ratio.
            ox_temp: Temperature of oxidizer inlet (K).
            fuel_temp: Temperature of fuel inlet (K).

        Returns:
            Tuple (b0_dict, h0_r_mix)
        """
        # 1. Get properties of reactants
        ox_spec = self.db.species.get(oxidizer)
        fuel_spec = self.db.species.get(fuel)

        if not ox_spec or not fuel_spec:
            raise ValueError(f"Reactants {oxidizer} or {fuel} not found in DB.")

        # 2. Calculate Enthalpies (H/RT -> H_spec [J/kg])
        # h_spec = (H/RT) * R_univ * T / MW

        # Oxidizer
        _, h_rt_ox, _, _ = ox_spec.get_properties(ox_temp)
        h_ox_per_kg = (h_rt_ox * R_UNIV * ox_temp) / ox_spec.molecular_weight

        # Fuel
        _, h_rt_fu, _, _ = fuel_spec.get_properties(fuel_temp)
        h_fu_per_kg = (h_rt_fu * R_UNIV * fuel_temp) / fuel_spec.molecular_weight

        # 3. Mixture Enthalpy (per kg)
        # H_mix = (m_ox * h_ox + m_fu * h_fu) / (m_ox + m_fu)
        # using m_ox = O/F, m_fu = 1
        total_mass = of_ratio + 1.0
        h_mix_per_kg = (of_ratio * h_ox_per_kg + 1.0 * h_fu_per_kg) / total_mass

        # Convert to solver input format: H0/R [K/kg] ? 
        # Solver expects Val_1 = H0/R (units of Temperature per unit mass basis? No.)
        # In equilibrium.py: h0_rt_target = val_1 / current_T
        # RHS Enthalpy term is dimensionless sum(nj * H/RT). nj is kmol/kg.
        # So sum(nj * H/RT) * RT = Enthalpy_J_per_kg.
        # Therefore, H0/R should be (Enthalpy_J_per_kg / R_univ).
        # Check units: [J/kg] / [J/kmol*K] = [kmol*K / kg].
        # Correct.
        h0_r_mix = h_mix_per_kg / R_UNIV

        # 4. Elemental Composition (b0: kmol_element / kg_mix)
        b0 = {}

        # Helper to add atoms
        def add_atoms(spec, mass_frac, temp_b0):
            moles_spec_per_kg = mass_frac / spec.molecular_weight  # kmol/kg
            for el, count in spec.composition.items():
                temp_b0[el] = temp_b0.get(el, 0.0) + count * moles_spec_per_kg

        add_atoms(ox_spec, of_ratio / total_mass, b0)
        add_atoms(fuel_spec, 1.0 / total_mass, b0)

        return b0, h0_r_mix


class RocketEngine:
    def __init__(self, thermo_path: str = 'thermo.inp'):
        print(f"Loading database from {thermo_path}...")
        self.db = ThermoDatabase()
        self.db.load_from_file(thermo_path)
        self.solver = EquilibriumSolver(self.db)
        self.propellant_helper = PropellantInput(self.db)

        # Storage for results
        self.chamber: Optional[EquilibriumResult] = None
        self.throat: Optional[EquilibriumResult] = None
        self.exit: Optional[EquilibriumResult] = None

        # Standard species list to consider for combustion products
        # In a real app, this might be dynamic or user-selected.
        self.candidates = [
            "H2", "O2", "H2O", "H", "O", "OH",  # H-O system
            "N2", "NO", "CO", "CO2", "CH4",  # C-N system
            # Add condensed phases if needed (names must match thermo.inp)
            # "H2O(L)", "H2O(s)" 
        ]

    def run_cycle(self,
                  pc_bar: float,
                  pe_bar: float,
                  oxidizer: str,
                  fuel: str,
                  of_ratio: float,
                  ox_temp: float = 90.0,  # Default LOX
                  fuel_temp: float = 20.0):  # Default LH2

        print("\n" + "=" * 50)
        print(f"ROCKET ENGINE SIMULATION: {oxidizer}/{fuel} (O/F={of_ratio})")
        print(f"Pc = {pc_bar} bar, Pe = {pe_bar} bar")
        print("=" * 50)

        # 1. Prepare Inputs
        b0, h0_r = self.propellant_helper.calculate_inputs(
            oxidizer, fuel, of_ratio, ox_temp, fuel_temp
        )

        # 2. Chamber (HP Problem)
        print("\n--- 1. CHAMBER CALCULATION (HP) ---")
        self.chamber = self.solver.solve(
            elements_b0=b0,
            init_species=self.candidates,
            P=pc_bar,
            val_1=h0_r,
            problem_type='HP'
        )

        if not self.chamber:
            print("Chamber calculation failed.")
            return

        self._print_state(self.chamber, "CHAMBER")

        # 3. Throat (SP Problem with Mach condition)
        print("\n--- 2. THROAT SEARCH (M=1) ---")
        # Entropy from chamber to be held constant
        s0_r = self.chamber.properties['s_total_r']

        # We need to find P_t such that velocity u == sonic_velocity a.
        # Conservation of Energy: H0 = H_t + u^2/2  => u = sqrt(2*(H0 - H_t))
        # Condition: sqrt(2*(H0 - H_t)) - a_t = 0

        h0_j_kg = self.chamber.properties['h_total_rt'] * R_UNIV * self.chamber.T / self.chamber.properties['mw']
        # Actually, simpler: H0/R is conserved (if adiabatic).
        # Let's use specific enthalpy in J/kg for velocity calc.
        H_stagnation = h0_j_kg

        def throat_residual(p_guess):
            # Solve SP at p_guess
            res = self.solver.solve(
                elements_b0=b0,
                init_species=self.candidates,  # Should ideally use results from chamber as init guess
                P=p_guess,
                val_1=s0_r,
                problem_type='SP'
            )
            if not res: return 1e9  # Penalty

            # Calculate Velocity u
            # H_t (J/kg)
            h_t = res.properties['h_total_rt'] * R_UNIV * res.T / res.properties['mw']

            delta_h = H_stagnation - h_t
            if delta_h < 0: return 1e9  # Physically impossible (heating up?)

            u = np.sqrt(2 * delta_h)
            a = res.properties['son_vel']

            return u - a

        # Search range for Throat Pressure: usually 0.5 * Pc to 0.6 * Pc
        try:
            pt_sol = brentq(throat_residual, 0.1 * pc_bar, 0.9 * pc_bar, rtol=1e-4)

            # Recalculate final throat state
            self.throat = self.solver.solve(
                elements_b0=b0,
                init_species=self.candidates,
                P=pt_sol,
                val_1=s0_r,
                problem_type='SP'
            )
            self._print_state(self.throat, "THROAT")

        except Exception as e:
            print(f"Throat search failed: {e}")
            return

        # 4. Exit (SP Problem)
        print(f"\n--- 3. EXIT CALCULATION (Pe={pe_bar} bar) ---")
        self.exit = self.solver.solve(
            elements_b0=b0,
            init_species=self.candidates,
            P=pe_bar,
            val_1=s0_r,
            problem_type='SP'
        )

        if self.exit:
            self._print_state(self.exit, "EXIT")
            self._print_performance(H_stagnation)

    def _print_state(self, res: EquilibriumResult, label: str):
        print(f"[{label}]")
        print(f"  P = {res.P:.4f} bar")
        print(f"  T = {res.T:.2f} K")
        props = res.properties
        print(f"  Gamma = {props.get('gamma_s', 0.0):.4f}")
        print(f"  Mw    = {props.get('mw', 0.0):.4f} kg/kmol")
        print(f"  Sonic = {props.get('son_vel', 0.0):.1f} m/s")
        print("  Top Species (>1%):")
        sorted_specs = sorted(res.mole_fractions.items(), key=lambda x: x[1], reverse=True)
        for name, frac in sorted_specs:
            if frac > 0.01:
                print(f"    {name}: {frac:.4f}")

    def _print_performance(self, H_stag_j_kg: float):
        if not self.throat or not self.exit: return

        print("\n--- PERFORMANCE SUMMARY ---")

        # 1. Exit Velocity
        h_e = self.exit.properties['h_total_rt'] * R_UNIV * self.exit.T / self.exit.properties['mw']
        u_exit = np.sqrt(2 * (H_stag_j_kg - h_e))

        # 2. Isp (Vacuum and Sea Level components usually separate, here ideal expansion)
        # Isp = u_e / g0
        isp = u_exit / 9.80665

        # 3. Characteristic Velocity (C*)
        # C* = P_c * A_t / m_dot
        # m_dot / A_t = rho_t * u_t
        # rho_t = P_t / ( (R/Mw) * T_t )  (Ideal gas approx for rho)
        # Better: rho derived from 1/V specific volume if we had it. 
        # Using P = rho * (R_univ/Mw) * T

        # Throat properties
        Pt_pa = self.throat.P * 1e5
        Tt = self.throat.T
        Mwt = self.throat.properties['mw']
        Rt_specific = R_UNIV / Mwt
        rhot = Pt_pa / (Rt_specific * Tt)
        ut = self.throat.properties['son_vel']

        flux_throat = rhot * ut  # kg/(s*m2)
        c_star = (self.chamber.P * 1e5) / flux_throat

        # 4. Expansion Ratio (Area Ratio)
        # Ae / At = (rho_t * u_t) / (rho_e * u_e)
        Pe_pa = self.exit.P * 1e5
        Te = self.exit.T
        Mwe = self.exit.properties['mw']
        Re_specific = R_UNIV / Mwe
        rhoe = Pe_pa / (Re_specific * Te)

        area_ratio = flux_throat / (rhoe * u_exit)

        print(f"  Isp (Ideal)  = {isp:.2f} s")
        print(f"  Velocity Ue  = {u_exit:.2f} m/s")
        print(f"  C* (C-star)  = {c_star:.2f} m/s")
        print(f"  Area Ratio   = {area_ratio:.2f}")


if __name__ == "__main__":
    # Example usage
    import os

    if not os.path.exists('thermo.inp'):
        print("Error: thermo.inp not found. Please upload it.")
    else:
        engine = RocketEngine('thermo.inp')

        # Run standard test: H2/O2 engine
        # SSME-like conditions: Pc=200 bar, Pe=1 bar, O/F=6.0
        engine.run_cycle(
            pc_bar=100.0,
            pe_bar=1.0,
            oxidizer="O2",
            fuel="H2",
            of_ratio=5.5
        )