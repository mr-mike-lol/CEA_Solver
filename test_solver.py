import unittest
import numpy as np
from thermo import Species, ThermoDatabase
from equilibrium import EquilibriumSolver


def convert_7_to_9(c7):
    """
    Конвертирует массив 7 коэффициентов (NASA old format) в 9 коэффициентов (NASA new format structure).
    Old (7): Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    New (9): Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
    Mapping:
      New[0] (T^-2) = 0
      New[1] (T^-1) = 0
      New[2] (T^0)  = Old[0]
      New[3] (T^1)  = Old[1]
      New[4] (T^2)  = Old[2]
      New[5] (T^3)  = Old[3]
      New[6] (T^4)  = Old[4]
      New[7] (H const) = Old[5]
      New[8] (S const) = Old[6]
    """
    c9 = [0.0, 0.0, c7[0], c7[1], c7[2], c7[3], c7[4], c7[5], c7[6]]
    return c9


class TestCombustion(unittest.TestCase):
    def setUp(self):
        self.db = ThermoDatabase()

        # --- ДАННЫЕ NASA 7-коэффициентов (High Temp 1000-5000K) ---
        # Взяты из базы данных Burcat / CEA (thermo.inp)

        # H2
        c_h2 = [2.9328305E+00, 8.2660803E-04, -1.4640237E-07, 1.5410041E-11, -6.8880443E-16, -8.1306558E+02,
                -1.0243288E+00]
        sp_h2 = Species("H2", {'H': 2}, 2.016)
        sp_h2.add_interval(1000, 6000, convert_7_to_9(c_h2))
        self.db.add_species(sp_h2)

        # O2
        c_o2 = [3.6219535E+00, 7.3618290E-04, -1.9652276E-07, 3.6202085E-11, -2.8893966E-15, -1.2039756E+03,
                3.6150373E+00]
        sp_o2 = Species("O2", {'O': 2}, 31.999)
        sp_o2.add_interval(1000, 6000, convert_7_to_9(c_o2))
        self.db.add_species(sp_o2)

        # H2O
        c_h2o = [2.6770389E+00, 2.9731816E-03, -7.7376889E-07, 9.4433514E-11, -4.2689991E-15, -2.9885894E+04,
                 6.8825500E+00]
        sp_h2o = Species("H2O", {'H': 2, 'O': 1}, 18.015)
        sp_h2o.add_interval(1000, 6000, convert_7_to_9(c_h2o))
        self.db.add_species(sp_h2o)

        # OH
        c_oh = [2.8385303E+00, 1.1074129E-03, -2.9400021E-07, 4.2093072E-11, -2.4229270E-15, 3.7004681E+03,
                5.8442231E+00]
        sp_oh = Species("OH", {'O': 1, 'H': 1}, 17.007)
        sp_oh.add_interval(1000, 6000, convert_7_to_9(c_oh))
        self.db.add_species(sp_oh)

        # H
        c_h = [2.5000000E+00, 0.0, 0.0, 0.0, 0.0, 2.5473660E+04, -4.4819177E-01]
        sp_h = Species("H", {'H': 1}, 1.008)
        sp_h.add_interval(1000, 6000, convert_7_to_9(c_h))
        self.db.add_species(sp_h)

        # O
        c_o = [2.5436370E+00, -2.7316248E-05, -4.1902952E-09, 4.9548184E-12, -1.3395372E-15, 2.9122259E+04,
               4.9015496E+00]
        sp_o = Species("O", {'O': 1}, 15.999)
        sp_o.add_interval(1000, 6000, convert_7_to_9(c_o))
        self.db.add_species(sp_o)

        self.solver = EquilibriumSolver(self.db)

    def test_h2_o2_combustion(self):
        # Вход: стехиометрия H2O (2 моль H, 1 моль O)
        b0 = {'H': 2.0, 'O': 1.0}
        species = ["H2", "O2", "H2O", "OH", "H", "O"]

        # P = 10 bar, T = 3200 K
        state = self.solver.solve(b0, species, 10.0, 3200.0)

        self.assertIsNotNone(state)

        print("\n=== РЕЗУЛЬТАТЫ РАСЧЕТА (H2+O2, T=3200K, P=10bar) ===")
        # Сортируем и печатаем
        sorted_res = sorted(state.n.items(), key=lambda x: x[1], reverse=True)
        for name, mol in sorted_res:
            frac = mol / state.total_moles
            print(f"{name}: {frac:.5f}")

        # Проверки физики
        # 1. Основной продукт - вода (>60%)
        self.assertGreater(state.n['H2O'] / state.total_moles, 0.60)

        # 2. Диссоциация должна быть! OH должен быть > 1%
        # При 3200К в равновесии около 3-5% OH
        self.assertGreater(state.n['OH'] / state.total_moles, 0.01)

        # 3. Атомарный водород тоже должен быть заметен
        self.assertGreater(state.n['H'] / state.total_moles, 0.001)


if __name__ == '__main__':
    unittest.main()