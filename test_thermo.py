import os
import sys
from thermo import ThermoDatabase


def test_species(db, species_name, temperature):
    """
    Выводит информацию о веществе и рассчитывает его свойства при заданной Т.
    """
    spec = db.get_species(species_name)

    print(f"--- Testing Species: {species_name} ---")
    if spec is None:
        print(f"ERROR: Species '{species_name}' not found in database!")
        return

    print(f"Composition: {spec.composition}")
    print(f"Molecular Weight: {spec.molecular_weight}")
    print(f"Number of Intervals: {len(spec.intervals)}")

    # Вывод коэффициентов для каждого интервала
    for idx, interval in enumerate(spec.intervals):
        print(f"  Interval {idx + 1}: {interval.t_min} K - {interval.t_max} K")
        print(f"  Coeffs ({len(interval.coeffs)}): {interval.coeffs}")

    # Расчет свойств
    try:
        cp_r, h_rt, s_r, g_rt = spec.get_properties(temperature)
        print(f"\nProperties at T = {temperature} K:")
        print(f"  Cp/R = {cp_r:.6f}")
        print(f"  H/RT = {h_rt:.6f}")
        print(f"  S/R  = {s_r:.6f}")
        print(f"  G/RT = {g_rt:.6f}")

        # Для проверки размерных величин (Дж/кг/К и т.д.)
        # R_univ = 8314.51 J/(kmol*K)
        # M = spec.molecular_weight (kg/kmol, если в базе старый формат, часто g/mol, надо проверить)
        # Обычно в thermo.inp вес в g/mol или kg/kmol (число одинаковое).
        # Но CEA использует SI, где M часто нужно делить на 1000 если переводим в кг?
        # Пока смотрим только безразмерные, они надежнее.

    except Exception as e:
        print(f"Error calculating properties: {e}")
    print("\n" + "=" * 30 + "\n")


def main():
    db = ThermoDatabase()

    # Проверяем наличие файла
    inp_file = 'thermo.inp'
    if not os.path.exists(inp_file):
        print(f"File '{inp_file}' not found. Please upload it.")
        return

    print(f"Loading database from {inp_file}...")
    db.load_from_file(inp_file)
    print(f"Database loaded. Total species: {len(db.species)}")
    print("=" * 30 + "\n")

    # Тестируем Водород и Кислород (Основные компоненты)
    # Имена должны точно совпадать с теми, что в файле.
    # Обычно это "H2" и "O2" (иногда "H2(g)" в специфичных базах, но стандарт CEA - просто формула).

    # Тест 1: Водород при 1000 К
    test_species(db, "H2", 1000.0)

    # Тест 2: Кислород при 1000 К
    test_species(db, "O2", 1000.0)

    # Тест 3: Вода (продукт реакции) при 3000 К (типичная Т в камере)
    test_species(db, "H2O", 3000.0)


if __name__ == "__main__":
    main()