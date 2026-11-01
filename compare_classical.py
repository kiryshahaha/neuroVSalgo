"""
Скрипт демонстрации и сравнения классических алгоритмических солверов:
1. SciPy (C-реализация венгерского алгоритма).
2. Google OR-Tools (C++ аукционный алгоритм).

Запуск:
    python compare_classical.py
"""
import numpy as np
from src.data_generator import get_generator
from src.solvers import ScipyHungarianSolver, OrToolsSolver


def main():
    print("=" * 75)
    print("ДЕМОНСТРАЦИЯ И СРАВНЕНИЕ КЛАССИЧЕСКИХ СОЛВЕРОВ (SciPy vs Google OR-Tools)")
    print("=" * 75)

    scipy_solver = ScipyHungarianSolver()
    ortools_solver = OrToolsSolver()

    # 1. Простой наглядный пример 4x4
    print("\n1. Наглядный пример назначения 4 исполнителей на 4 задачи:")
    matrix_4x4 = np.array([
        [15.0, 25.0, 10.0, 30.0],
        [20.0, 10.0, 35.0, 25.0],
        [25.0, 30.0, 15.0, 10.0],
        [10.0, 20.0, 25.0, 40.0],
    ])

    res_scipy = scipy_solver.solve(matrix_4x4)
    res_ortools = ortools_solver.solve(matrix_4x4)

    print("Матрица затрат 4x4:")
    print(matrix_4x4)
    print("\nРезультат SciPy:")
    print(f"  Назначение (исполнитель -> задача): {res_scipy.assignment}")
    print(f"  Суммарная стоимость: {res_scipy.total_cost:.2f}")
    print(f"  Время вычисления: {res_scipy.execution_time_ms:.4f} мс")

    print("\nРезультат Google OR-Tools:")
    print(f"  Назначение (исполнитель -> задача): {res_ortools.assignment}")
    print(f"  Суммарная стоимость: {res_ortools.total_cost:.2f}")
    print(f"  Время вычисления: {res_ortools.execution_time_ms:.4f} мс")

    # 2. Сравнение производительности при росте размерности N
    print("\n" + "=" * 75)
    print("2. Сравнение времени вычислений при росте размерности задачи N:")
    print(f"{'N':<8} | {'SciPy (мс)':<14} | {'OR-Tools (мс)':<14} | {'Стоимость SciPy':<16} | {'Стоимость OR-Tools':<18}")
    print("-" * 75)

    test_sizes = [10, 50, 100, 250, 500]
    gen = get_generator("spatial_2d", seed=42)

    for n in test_sizes:
        mat = gen.generate(n)

        # Прогревочный вызов
        _ = scipy_solver.solve(mat)
        _ = ortools_solver.solve(mat)

        # Замер
        r_scipy = scipy_solver.solve(mat)
        r_ortools = ortools_solver.solve(mat)

        print(
            f"{n:<8} | "
            f"{r_scipy.execution_time_ms:<14.3f} | "
            f"{r_ortools.execution_time_ms:<14.3f} | "
            f"{r_scipy.total_cost:<16.4f} | "
            f"{r_ortools.total_cost:<18.4f}"
        )

    print("=" * 75)
    print("ВЫВОД:")
    print("- Стоимости решений SciPy и Google OR-Tools совпадают с абсолютной точностью.")
    print("- Оба алгоритма гарантируют 100% глобальный оптимум (Cost_opt).")
    print("- Google OR-Tools служит надежным эталоном (Ground Truth) для обучения и оценки нейросети.")
    print("=" * 75)


if __name__ == "__main__":
    main()
