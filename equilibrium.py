import numpy as np
import logging
from math import exp, log

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("CEA_Solver")


class EquilibriumState:
    def __init__(self, T, P):
        self.T = T
        self.P = P
        self.n = {}
        self.total_moles = 0.0
        self.M = 0.0


class EquilibriumSolver:
    def __init__(self, thermo_db):
        self.db = thermo_db
        self.MAX_ITER = 100
        self.TOLERANCE = 1e-6
        self.trace_limit = 1e-12

    def solve(self, elements_b0, init_species, P, T_guess, problem_type='TP'):
        logger.info(f"Solving {problem_type} at P={P} bar, T={T_guess} K")

        # 1. Подготовка
        species_objs = [self.db.species[name] for name in init_species if name in self.db.species]
        n_species = len(species_objs)
        element_list = list(elements_b0.keys())
        n_elements = len(element_list)

        # Матрица состава a[i, j] (элемент i в веществе j)
        a = np.zeros((n_elements, n_species))
        for j, sp in enumerate(species_objs):
            for i, el in enumerate(element_list):
                a[i, j] = sp.composition.get(el, 0.0)

        b0 = np.array([elements_b0[el] for el in element_list])

        # Улучшенное начальное приближение
        # Масштабируем количество молей так, чтобы грубо соблюсти баланс масс
        # n_j ~ b0_i / a_ij (очень грубо)
        nj = np.ones(n_species) * 0.1
        # Попытка нормализации: пусть суммарная масса элементов совпадает
        calc_b = np.dot(a, nj)
        scale_factor = np.mean(b0 / (calc_b + 1e-9))
        nj *= scale_factor

        ln_nj = np.log(nj)
        total_n = np.sum(nj)
        ln_total_n = np.log(total_n)

        current_T = T_guess

        for iteration in range(self.MAX_ITER):
            # 2. Термодинамика
            mu_rt = np.zeros(n_species)
            for j, sp in enumerate(species_objs):
                _, h_rt, s_r, _ = sp.get_properties(current_T)
                # mu/RT = H/RT - S/R + ln(nj) + ln(P/N)
                mu_rt[j] = h_rt - s_r + ln_nj[j] + np.log(P) - ln_total_n

            # 3. Сборка матрицы (TP problem: size Nel + 1)
            dim = n_elements + 1
            matrix = np.zeros((dim, dim))
            rhs = np.zeros(dim)

            # Блоки матрицы
            # A_ik = sum(a_ij * a_kj * nj)
            # Это можно записать матрично: A @ diag(nj) @ A.T
            A_matrix = a
            nj_diag = np.diag(nj)

            # Верхний левый блок (Elements)
            matrix[:n_elements, :n_elements] = A_matrix @ nj_diag @ A_matrix.T

            # Столбец/строка суммы молей
            # A_in = sum(a_ij * nj)
            vec_n = A_matrix @ nj
            matrix[:n_elements, n_elements] = vec_n
            matrix[n_elements, :n_elements] = vec_n

            # Нижний правый элемент (Sum moles)
            matrix[n_elements, n_elements] = np.sum(nj) - total_n

            # Правая часть (Residuals)
            # Элементы: b0 - sum(a_ij * nj) + sum(a_ij * nj * mu)
            # Векторно: b0 - vec_n + A @ (nj * mu)
            term_mu = nj * mu_rt
            rhs[:n_elements] = b0 - vec_n + A_matrix @ term_mu

            # Сумма молей: N - sum(nj) + sum(nj * mu)
            rhs[n_elements] = total_n - np.sum(nj) + np.sum(term_mu)

            # 4. Решение
            try:
                x = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                # Попытка регуляризации при сингулярности
                logger.warning("Singular matrix, adding regularization")
                matrix += np.eye(dim) * 1e-9
                try:
                    x = np.linalg.solve(matrix, rhs)
                except:
                    return None

            pi = x[:n_elements]
            dln_total_n = x[n_elements]

            # Поправки
            # dln_nj = -mu + A.T @ pi + dln_N
            dln_nj = -mu_rt + A_matrix.T @ pi + dln_total_n

            # Lambda control
            lambda_f = 1.0
            max_dln = np.max(np.abs(dln_nj))
            if max_dln > 2.0:
                lambda_f = 2.0 / max_dln

            # Применение
            ln_nj += lambda_f * dln_nj
            ln_total_n += lambda_f * dln_total_n

            nj = np.exp(ln_nj)
            total_n = np.exp(ln_total_n)

            if max_dln * lambda_f < self.TOLERANCE:
                logger.info(f"Converged in {iteration} iterations")
                res = EquilibriumState(current_T, P)
                for j, sp in enumerate(species_objs):
                    if nj[j] > self.trace_limit:
                        res.n[sp.name] = nj[j]
                res.total_moles = total_n
                return res

        logger.error("Max iterations reached")
        return None