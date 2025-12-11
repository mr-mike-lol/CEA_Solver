import numpy as np
from scipy.optimize import minimize_scalar, brentq

from equilibrium import EquilibriumSolver, Reactant
from thermo import ThermoDatabase
from constants import R_UNIV


class RocketEngine:
    def __init__(self, thermo_file='thermo.inp'):
        self.db = ThermoDatabase()
        self.db.load(thermo_file)
        self.solver = EquilibriumSolver(self.db)

        # Результаты
        self.results = {}

    def _get_state(self):
        """
        Извлекает текущее состояние из решателя и рассчитывает газодинамические параметры.
        """
        T = self.solver.T
        P = self.solver.P * 1e5  # bar -> Pa

        # 1. Свойства смеси
        H_total = self.solver.get_enthalpy_mixture()  # J
        S_total = self.solver.get_entropy_mixture()  # J/K
        Cp_total = self.solver.get_heat_capacity_mixture()  # J/K

        # 2. Масса системы
        mass_g = 0.0
        for i, nj in enumerate(self.solver.n):
            mw = self.solver.product_species[i].molecular_weight
            mass_g += nj * mw
        mass_kg = mass_g / 1000.0

        # 3. Удельные свойства (на кг)
        h = H_total / mass_kg
        s = S_total / mass_kg
        cp = Cp_total / mass_kg

        # Молярная масса
        MW_mix_g_mol = self.solver.get_mixture_molecular_weight()

        # Газовая постоянная
        R_spec = R_UNIV / MW_mix_g_mol * 1000.0  # J/(kg*K)

        # Gamma
        # Если Cp < R (что физически невозможно, но бывает при ошибках), ставим заглушку
        if cp <= R_spec:
            gamma = 1.1
        else:
            gamma = cp / (cp - R_spec)

        # Скорость звука
        a_sound = np.sqrt(gamma * R_spec * T)

        return {
            'P': P, 'T': T, 'H': h, 'S': s, 'Cp': cp,
            'MW': MW_mix_g_mol, 'gamma': gamma,
            'a': a_sound,
            'rho': P / (R_spec * T)
        }

    def solve_expansion(self, S_target, P_target_bar):
        """
        Isentropic Expansion: Найти T, при которой S(T, P_target) == S_target.
        """

        # Сброс состава решателя на "грубую" прикидку, чтобы избежать застревания в локальных минимумах
        # (Опционально, если есть метод reset)

        def entropy_residual(T_guess):
            # Защита от отрицательных T
            if T_guess < 200: T_guess = 200

            # Решаем равновесие
            success = self.solver.solve_tp(float(T_guess), P_target_bar)
            # Если не сошлось, это плохо, но попробуем вернуть результат как есть
            # или можно вернуть +inf

            state = self._get_state()
            return state['S'] - S_target

        # Границы поиска T.
        # Верхняя: T в камере + немного (хотя при расширении T падает).
        # Нижняя: 600 K (чтобы не улететь в зону плохой сходимости).
        T_max = self.solver.T + 200.0
        T_min = 400.0

        # Проверка границ перед запуском brentq
        try:
            s_min = entropy_residual(T_min)
            s_max = entropy_residual(T_max)

            if s_min * s_max > 0:
                # Знаки одинаковые — корень не внутри.
                # Обычно S(T_min) < S_target < S(T_max).
                # S_residual = S_calc - S_target.
                # Значит S(T_min) - S_target < 0, S(T_max) - S_target > 0.

                # Если оба > 0: Значит S(T_min) уже > S_target. Температура должна быть ЕЩЕ ниже.
                if s_min > 0:
                    # print(f"Warning: Low temp limit reached at {P_target_bar} bar. T_sol < 400K.")
                    return None  # Слишком холодно, выходим

                # Если оба < 0: S(T_max) < S_target. Странно, T должна быть выше T_chamber? Невозможно при расширении.
                if s_max < 0:
                    return None

            T_sol = brentq(entropy_residual, T_min, T_max)
            self.solver.solve_tp(T_sol, P_target_bar)
            return self._get_state()

        except Exception as e:
            # print(f"Expansion error to {P_target_bar} bar: {e}")
            return None

    def find_throat(self, chamber_state):
        S_target = chamber_state['S']
        H_c = chamber_state['H']
        P_c = chamber_state['P'] / 1e5  # bar

        def throat_error(P_guess_bar):
            state = self.solve_expansion(S_target, P_guess_bar)
            if not state:
                # Если не удалось посчитать, возвращаем штраф
                # Пытаемся понять, в какую сторону двигаться
                return 1e6

                # u = sqrt(2*dH)
            delta_H = H_c - state['H']
            if delta_H < 0: delta_H = 0
            u = np.sqrt(2 * delta_H)

            # Нам нужно u - a = 0
            return u - state['a']

        # Сканируем диапазон давлений, чтобы найти пересечение нуля
        # Вместо слепого brentq, пройдемся сеткой, чтобы найти интервал смены знака
        p_scan = np.linspace(P_c * 0.95, P_c * 0.1, 10)  # От камеры вниз
        vals = []
        for p in p_scan:
            vals.append(throat_error(p))

        # Ищем смену знака
        p_a, p_b = None, None
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0 and abs(vals[i]) < 1e5:
                p_a = p_scan[i]
                p_b = p_scan[i + 1]
                break

        if p_a is None:
            print("   Warning: Could not bracket throat pressure (M=1).")
            # print(f"   Debug vals: {vals}")
            return None

        try:
            P_throat_bar = brentq(throat_error, p_a, p_b)
            self.solve_expansion(S_target, P_throat_bar)
            throat_state = self._get_state()
            throat_state['u'] = throat_state['a']
            return throat_state
        except Exception as e:
            print(f"Throat solver failed: {e}")
            return None

    def run(self, P_chamber_bar, P_exit_bar, reactants):
        print(f"--- Rocket Calculation (Pc={P_chamber_bar} bar, Pe={P_exit_bar} bar) ---")

        # 1. Combustion
        print("1. Solving Combustion Chamber...")
        self.solver.set_reactants(reactants)
        if not self.solver.solve_hp(P_chamber_bar):
            print("Combustion failed.")
            return

        self.chamber = self._get_state()
        self.chamber['u'] = 0.0
        print(f"   Tc = {self.chamber['T']:.1f} K, P = {P_chamber_bar:.1f} bar")

        # 2. Throat
        print("2. Searching for Throat (M=1)...")
        self.throat = self.find_throat(self.chamber)

        if self.throat:
            # C* = Pc * At / m_dot = Pc / (rho*u)_t
            flux_throat = self.throat['rho'] * self.throat['u']
            c_star = self.chamber['P'] / flux_throat
            print(f"   Tt = {self.throat['T']:.1f} K, Pt = {self.throat['P'] / 1e5:.2f} bar")
            print(f"   C* = {c_star:.1f} m/s")
        else:
            print("   Throat search failed. Continuing with Exit only...")

        # 3. Exit
        print(f"3. Expanding to Exit (Pe={P_exit_bar} bar)...")
        self.exit = self.solve_expansion(self.chamber['S'], P_exit_bar)

        if self.exit:
            u_exit = np.sqrt(2 * (self.chamber['H'] - self.exit['H']))
            self.exit['u'] = u_exit
            isp = u_exit / 9.80665

            print(f"   Te = {self.exit['T']:.1f} K, Ue = {u_exit:.1f} m/s")
            print(f"   Isp = {isp:.1f} s")

            if self.throat:
                flux_exit = self.exit['rho'] * u_exit
                area_ratio = flux_throat / flux_exit
                print(f"   Area Ratio = {area_ratio:.2f}")

        # Composition output
        print("\nExit Composition:")
        self.solver.print_results()


if __name__ == "__main__":
    eng = RocketEngine()

    # Тест: H2 + O2, O/F = 4 (Богатая)
    # H2=2, O2=32.
    # 1 mol O2 (32g) + 8 mol H2 (16g) -> O/F = 2.0 (Очень богатая)
    # 1 mol O2 + 2 mol H2 -> O/F = 8.0 (Стехиометрия)
    # Попробуем близко к стехиометрии для высокой температуры
    reacts = [
        Reactant("H2", moles=3.0, temp=200.0),
        Reactant("O2", moles=1.0, temp=90.0)
    ]

    eng.run(P_chamber_bar=100.0, P_exit_bar=1.0, reactants=reacts)