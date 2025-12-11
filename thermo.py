import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Константа из исходного кода
R_UNIV = 8314.51


def parse_double(text: str) -> float:
    """
    Конвертирует Fortran-формат числа (1.234D+02) в Python float.
    Обрабатывает замену D на E и пробелы.
    """
    if not text or text.isspace():
        return 0.0
    try:
        # Замена D на E для экспоненты
        clean_text = text.replace('D', 'E').replace('d', 'e').strip()
        return float(clean_text)
    except ValueError:
        return 0.0


@dataclass
class Nasa9Poly:
    """
    Хранит коэффициенты полинома NASA (9 коэффициентов) для заданного диапазона температур.
    """
    t_min: float
    t_max: float
    coeffs: np.ndarray

    def calc_properties(self, T: float) -> Tuple[float, float, float]:
        """
        Рассчитывает Cp/R, H/RT и S/R для заданной температуры.
        Формулы взяты из подпрограммы CPHS.
        [cite: 30-34]
        """
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T

        a = self.coeffs

        # Cp/R
        cp_r = (a[0] / T2) + (a[1] / T) + a[2] + (a[3] * T) + (a[4] * T2) + (a[5] * T3) + (a[6] * T4)

        # H/RT
        h_rt = (-a[0] / T2) + (a[1] * np.log(T) / T) + a[2] + (a[3] * T / 2.0) + \
               (a[4] * T2 / 3.0) + (a[5] * T3 / 4.0) + (a[6] * T4 / 5.0) + (a[7] / T)

        # S/R
        s_r = (-a[0] / (2.0 * T2)) - (a[1] / T) + (a[2] * np.log(T)) + (a[3] * T) + \
              (a[4] * T2 / 2.0) + (a[5] * T3 / 3.0) + (a[6] * T4 / 4.0) + a[8]

        return cp_r, h_rt, s_r


@dataclass
class Species:
    name: str
    phase: str
    composition: Dict[str, float]
    molecular_weight: float
    heat_of_formation: float
    polys: List[Nasa9Poly] = field(default_factory=list)

    def get_properties(self, T: float) -> Tuple[float, float, float]:
        # Поиск подходящего диапазона
        selected_poly = None
        for poly in self.polys:
            if poly.t_min <= T <= poly.t_max:
                selected_poly = poly
                break

        # Экстраполяция (берем ближайший)
        if selected_poly is None:
            distances = [min(abs(T - p.t_min), abs(T - p.t_max)) for p in self.polys]
            selected_poly = self.polys[np.argmin(distances)]

        return selected_poly.calc_properties(T)


class ThermoDatabase:
    """
    Парсер данных из файла thermo.inp (формат NASA CEA).
    """

    def __init__(self):
        self.species: Dict[str, Species] = {}

    def load(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self._parse_lines(lines)

    def _parse_lines(self, lines: List[str]):
        i = 0
        total_lines = len(lines)

        # 1. Пропускаем глобальный заголовок файла
        # Ищем первую строку, которая похожа на имя вещества (не начинается с числа, !, thermo)
        while i < total_lines:
            line = lines[i].strip()
            # Пропуск пустых строк, комментариев и заголовка "thermo"
            if not line or line.startswith('!') or line.lower().startswith('thermo'):
                i += 1
                continue
            # Пропуск строки с температурами (начинается с числа)
            if line[0].isdigit():
                i += 1
                continue
            break

        # 2. Основной цикл чтения веществ
        while i < total_lines:
            line = lines[i]

            # Проверка на конец файла
            if 'END' in line.upper()[:5]:
                break

            # --- Строка 1: Имя вещества ---
            # Критерий имени: начинается не с пробела и не с цифры (обычно)
            # Исключение: e- может быть сложным, но обычно thermo.inp форматирован жестко.
            if len(line) < 2 or line[0].isdigit() or line.startswith('!'):
                i += 1  # Пропуск мусора между блоками
                continue

            species_name = line[0:15].strip()
            # Дополнительная защита: если имя пустое, это не начало блока
            if not species_name:
                i += 1
                continue

            # --- Строка 2: Метаданные ---
            i += 1
            if i >= total_lines: break
            meta_line = lines[i]

            # Проверяем, действительно ли это метаданные (должны начинаться с числа интервалов)
            try:
                # Количество интервалов (первые 2 символа)
                num_intervals_str = meta_line[0:2].strip()
                if not num_intervals_str:
                    # Возможно, рассинхрон. Откатываемся и ищем следующее имя
                    continue
                num_intervals = int(num_intervals_str)
            except ValueError:
                # Если это не число, значит предыдущая строка не была именем.
                # Пропускаем ту строку и пробуем эту как имя.
                # (i уже увеличен, но цикл while начнется с i, который указывает на текущую meta_line)
                # Мы просто делаем continue, но нам нужно вернуться на шаг назад логически,
                # но проще просто продолжить сканирование.
                continue

            try:
                # Парсинг состава
                composition = {}
                idx = 10
                for _ in range(5):
                    if idx + 8 > len(meta_line): break
                    elem = meta_line[idx:idx + 2].strip()
                    count_str = meta_line[idx + 2:idx + 8].strip()
                    if elem and count_str and elem != '0':
                        composition[elem] = parse_double(count_str)
                    idx += 8

                # Фаза (0-газ, >0-конденсат)
                phase_code_str = meta_line[50:52].strip()
                phase = 'G'
                if phase_code_str and int(phase_code_str) > 0:
                    phase = 'C'  # Condensed

                # MW и Hf
                mw_str = meta_line[52:65]
                hf_str = meta_line[65:80]
                molecular_weight = parse_double(mw_str)
                heat_of_formation = parse_double(hf_str)

                current_species = Species(
                    name=species_name,
                    phase=phase,
                    composition=composition,
                    molecular_weight=molecular_weight,
                    heat_of_formation=heat_of_formation
                )

                # --- Чтение интервалов и коэффициентов ---
                # Формат NASA 9-coef:
                # Для каждого интервала:
                # Строка A: Температурный диапазон (2F11.3) + кол-во коэфф (i1)
                # Строка B: 5 коэффициентов (5D16.8)
                # Строка C: 4 коэффициента (2D16.8 + ...)

                i += 1  # Переход к первому интервалу

                for _ in range(num_intervals):
                    if i >= total_lines: break

                    # 1. Диапазон температур
                    range_line = lines[i]
                    t_min = parse_double(range_line[0:11])
                    t_max = parse_double(range_line[11:22])

                    # 2. Коэффициенты (строка 1)
                    i += 1
                    line_c1 = lines[i]
                    coeffs_list = []
                    # 5 чисел по 16 символов
                    for k in range(5):
                        start = k * 16
                        val = parse_double(line_c1[start: start + 16])
                        coeffs_list.append(val)

                    # 3. Коэффициенты (строка 2)
                    i += 1
                    line_c2 = lines[i]
                    # Первые 2 числа
                    coeffs_list.append(parse_double(line_c2[0:16]))  # a6
                    coeffs_list.append(parse_double(line_c2[16:32]))  # a7
                    # a8 и a9 часто сдвинуты (позиции 48 и 64, или просто 3 и 4 блок)
                    # В исходном коде: 2D16.8, 16x, 2D16.8. То есть пропуск 16 символов посередине.
                    # Позиции: 0-16, 16-32, (32-48 skip), 48-64, 64-80
                    coeffs_list.append(parse_double(line_c2[48:64]))  # a8
                    coeffs_list.append(parse_double(line_c2[64:80]))  # a9

                    # Создаем полином
                    poly = Nasa9Poly(t_min, t_max, np.array(coeffs_list))
                    current_species.polys.append(poly)

                    i += 1  # Готовимся к следующему интервалу или веществу

                # Успешно распарсили вещество
                self.species[species_name] = current_species

            except Exception as e:
                print(f"Warning: Failed to parse species block starting at '{species_name}': {e}")
                # Пытаемся восстановиться, пропуская строки до следующего вероятного начала блока
                # i уже увеличен, цикл while продолжит сканирование
                continue

    def get_species(self, name: str) -> Optional[Species]:
        return self.species.get(name)


if __name__ == "__main__":
    import os

    db = ThermoDatabase()

    # Проверка на наличие файла
    inp_file = "thermo.inp"
    if os.path.exists(inp_file):
        print(f"Loading {inp_file}...")
        db.load(inp_file)
        print(f"Loaded {len(db.species)} species.")

        # Тест для O2
        o2 = db.get_species("O2")
        if o2:
            print(f"\nExample Species: {o2.name}")
            print(f"Phase: {o2.phase}")
            print(f"MW: {o2.molecular_weight}")
            print(f"H_f: {o2.heat_of_formation}")
            # Проверка свойств при 1000 K
            cp, h, s = o2.get_properties(1000.0)
            print(f"At 1000K -> Cp/R: {cp:.4f}, H/RT: {h:.4f}, S/R: {s:.4f}")
        else:
            print("O2 not found in database.")
    else:
        print(f"File {inp_file} not found. Please provide the file.")