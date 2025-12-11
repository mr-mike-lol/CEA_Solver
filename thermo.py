import numpy as np


class ThermoInterval:
    """
    Хранит коэффициенты для одного температурного интервала.
    Соответствует логике хранения данных в массиве Coef в Fortran[cite: 259].
    """

    def __init__(self, t_min, t_max, coeffs):
        self.t_min = t_min
        self.t_max = t_max
        # coeffs: массив из 9 чисел [a1, a2, a3, a4, a5, a6, a7, b1, b2]
        # a1..a7 - для Cp
        # b1 (index 7) - константа интегрирования для H
        # b2 (index 8) - константа интегрирования для S
        self.coeffs = np.array(coeffs, dtype=np.float64)


class Species:
    """
    Представляет химическое вещество (Species).
    Хранит имя, состав и набор температурных интервалов.
    """

    def __init__(self, name, composition, molecular_weight):
        self.name = name
        # Словарь состава, например: {'H': 2, 'O': 1} для воды
        self.composition = composition
        self.molecular_weight = molecular_weight
        self.intervals = []  # Список объектов ThermoInterval

    def add_interval(self, t_min, t_max, coeffs):
        self.intervals.append(ThermoInterval(t_min, t_max, coeffs))
        # Сортируем интервалы по температуре для быстрого поиска
        self.intervals.sort(key=lambda x: x.t_min)

    def get_properties(self, T):
        """
        Вычисляет безразмерные Cp/R, H/RT, S/R и G/RT для заданной температуры T.
        Реализация логики SUBROUTINE CPHS.
        """
        # 1. Находим нужный интервал
        interval = None
        for i in self.intervals:
            if i.t_min <= T <= i.t_max:
                interval = i
                break

        # Если T немного выходит за границы (экстраполяция или ошибки округления),
        # берем крайние интервалы, как это подразумевается в поиске
        if interval is None:
            if T < self.intervals[0].t_min:
                interval = self.intervals[0]
            elif T > self.intervals[-1].t_max:
                interval = self.intervals[-1]
            else:
                # Попали в "дырку" между интервалами (редко, но бывает)
                # Берем ближайший верхний
                for i in self.intervals:
                    if T < i.t_max:
                        interval = i
                        break

        # Коэффициенты: a1..a7, b1, b2
        c = interval.coeffs

        # Предварительные вычисления степеней T
        # В Fortran это массив cx
        t_inv2 = T ** -2  # T^-2
        t_inv = 1.0 / T  # T^-1
        t = T  # T^1
        t2 = T ** 2  # T^2
        t3 = T ** 3  # T^3
        t4 = T ** 4  # T^4
        ln_t = np.log(T)

        # 2. Расчет Cp/R (Теплоемкость)
        # Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
        # См. [cite: 34]
        cp_r = (c[0] * t_inv2 +
                c[1] * t_inv +
                c[2] +
                c[3] * t +
                c[4] * t2 +
                c[5] * t3 +
                c[6] * t4)

        # 3. Расчет H/RT (Энтальпия)
        # H/RT = -a1*T^-2 + a2*T^-1*lnT + a3 + a4*T/2 + a5*T^2/3 + a6*T^3/4 + a7*T^4/5 + b1/T
        # См.
        h_rt = (-c[0] * t_inv2 +
                c[1] * t_inv * ln_t +
                c[2] +
                c[3] * t / 2.0 +
                c[4] * t2 / 3.0 +
                c[5] * t3 / 4.0 +
                c[6] * t4 / 5.0 +
                c[7] * t_inv)

        # 4. Расчет S/R (Энтропия)
        # S/R = -a1*T^-2/2 - a2*T^-1 + a3*lnT + a4*T + a5*T^2/2 + a6*T^3/3 + a7*T^4/4 + b2
        # См.
        s_r = (-c[0] * t_inv2 / 2.0 -
               c[1] * t_inv +
               c[2] * ln_t +
               c[3] * t +
               c[4] * t2 / 2.0 +
               c[5] * t3 / 3.0 +
               c[6] * t4 / 4.0 +
               c[8])

        # 5. Энергия Гиббса G/RT = H/RT - S/R
        # Это используется для минимизации
        g_rt = h_rt - s_r

        return cp_r, h_rt, s_r, g_rt


class ThermoDatabase:
    """
    Класс для загрузки и хранения базы данных термодинамики.
    Аналог функционала UTHERM.
    """

    def __init__(self):
        self.species = {}  # Словарь {name: Species}

    def load_from_file(self, filename):
        """
        Парсит файл формата thermo.inp (стандарт NASA CEA).
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Пропускаем комментарии и заголовок
            if line.strip().startswith('!') or 'THERMO' in line:
                i += 1
                continue

            # Конец файла
            if 'END' in line.upper():
                break

            # Читаем первую строку записи вещества (имя, формула, вес)
            # Формат фиксированный, см. UTHERM logic [cite: 821]
            # Пример: O2                O  2.00    0.00  0.00  0.00 0   200.000  6000.000
            name = line[0:15].strip()

            # Если имя пустое или служебное, пропускаем
            if not name or len(line) < 79:
                i += 1
                continue

            # Парсинг состава (4 пары: Элемент-Количество)
            # Колонки: 24-28, 29-33 и т.д.
            composition = {}
            for pos in range(24, 44, 5):
                elem = line[pos:pos + 2].strip()
                try:
                    count_str = line[pos + 2:pos + 5].strip()
                    if not count_str: continue
                    count = float(count_str)
                    if elem and count > 0:
                        composition[elem] = count
                except ValueError:
                    continue

            # Фаза (G - газ, C - конденсат). В CEA это позиция 44 (или рядом)
            # В данном простом парсере опустим сложную логику фаз пока что.

            # Температурные диапазоны (обычно t_low, t_high, t_mid)
            # Позиции 46-55, 56-65, 66-75
            try:
                t_low = float(line[45:55])
                t_high = float(line[55:65])
                t_mid = float(line[65:75])
            except ValueError:
                # Если не удалось считать, пропускаем блок
                i += 1
                continue

            # Молекулярный вес (обычно вычисляется, но в файле он есть в конце строки или в след блоке)
            # В старом формате CEA он может быть в конце первой строки, но для простоты
            # здесь мы его пока поставим 0, его надо брать из constants или вычислять.
            mw = 0.0

            current_species = Species(name, composition, mw)

            # Читаем коэффициенты. Обычно 3 строки по 5 чисел.
            # 1-й интервал (High temp: t_mid -> t_high)
            # 2-й интервал (Low temp: t_low -> t_mid)
            # Стандарт NASA: сначала идут коэффициенты для ВЕРХНЕГО интервала, потом для НИЖНЕГО.

            coeffs_lines = lines[i + 1:i + 4]
            i += 4  # Переходим к следующему веществу

            # Собираем все числа из 3 строк
            raw_nums = []
            for cl in coeffs_lines:
                # Каждое число занимает 15 символов
                for k in range(0, 75, 15):
                    val_str = cl[k:k + 15]
                    if val_str.strip():
                        raw_nums.append(float(val_str))

            # Проверка целостности (должно быть 14 или 15 чисел, NASA использует 7+2=9 на интервал)
            # В файле thermo.inp обычно 14 коэффициентов (7 для high, 7 для low) + 2 константы интегрирования?
            # Нет, формат NASA-9 полиномов сложнее.
            # Стандартный "старый" формат NASA-7 (который часто встречается):
            # 5 чисел на строке 1
            # 5 чисел на строке 2
            # 4 числа на строке 3
            # Итого 14 чисел.
            # Первые 7 чисел: для T_mid -> T_high (a1..a7) - ОШИБКА, в старом формате a1..a5, a6, a7
            # Точный формат 7-коэфф (стандартный thermo.inp):
            # Строки 2-4 содержат коэффициенты.
            # Первые 7 чисел: a1..a7 для ВЕРХНЕГО интервала (t_mid -> t_high).
            # Вторые 7 чисел: a1..a7 для НИЖНЕГО интервала (t_low -> t_mid).
            # Формула CEA: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4.
            # H/RT = a1 + a2/2*T + ... + a6/T
            # S/R = a1*lnT + a2*T ... + a7

            # ВНИМАНИЕ: Ваш файл py_cea.f использует 9-коэффициентную формулу (T^-2, T^-1...).
            # Это современный формат NASA (2002+).
            # Если вы используете 'thermo.inp' старого формата (7 коэфф), математика Species.get_properties не совпадет.
            # Я написал код выше для 9 коэффициентов (как в py_cea.f).
            # Для запуска нам нужно убедиться, что у вас есть файл thermo.inp нового формата
            # или адаптировать код под старый (7 коэфф).

            # Предположим пока, что мы читаем данные вручную или создаем их,
            # так как парсинг фиксированного формата Fortran на Python — это отдельная большая задача.
            # Чтобы код работал "прямо сейчас", мы будем добавлять вещества вручную в main.

            self.species[name] = current_species

    def add_species(self, species):
        self.species[species.name] = species