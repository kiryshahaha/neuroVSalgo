import unittest
import numpy as np

from src.data_generator import UniformGenerator, Spatial2DGenerator
from src.solvers import ScipyHungarianSolver, OrToolsSolver


class TestClassicalSolvers(unittest.TestCase):
    def setUp(self):
        self.scipy_solver = ScipyHungarianSolver()
        self.ortools_solver = OrToolsSolver()

    def test_toy_matrix_known_optimum(self):
        # Известная матрица 3x3:
        # Исполнители: 0, 1, 2
        # Задачи:      0, 1, 2
        # [10, 20, 30]
        # [30, 10, 20]
        # [20, 30, 10]
        # Оптимальное назначение: (0->0, 1->1, 2->2), сумма = 10 + 10 + 10 = 30
        cost_matrix = np.array([
            [10.0, 20.0, 30.0],
            [30.0, 10.0, 20.0],
            [20.0, 30.0, 10.0]
        ], dtype=np.float64)

        res_scipy = self.scipy_solver.solve(cost_matrix)
        res_ortools = self.ortools_solver.solve(cost_matrix)

        self.assertTrue(res_scipy.is_feasible)
        self.assertTrue(res_ortools.is_feasible)

        self.assertAlmostEqual(res_scipy.total_cost, 30.0, places=5)
        self.assertAlmostEqual(res_ortools.total_cost, 30.0, places=5)

        np.testing.assert_array_equal(res_scipy.assignment, np.array([0, 1, 2]))
        np.testing.assert_array_equal(res_ortools.assignment, np.array([0, 1, 2]))

    def test_equivalence_on_random_matrices(self):
        """
        Ключевой математический тест: SciPy и Google OR-Tools ОБЯЗАНЫ
        давать идентичную суммарную стоимость на любых случайных матрицах.
        """
        gen = UniformGenerator(low=0.1, high=100.0, seed=777)
        sizes = [5, 15, 30, 60]

        for n in sizes:
            matrix = gen.generate(n)
            res_scipy = self.scipy_solver.solve(matrix)
            res_ortools = self.ortools_solver.solve(matrix)

            self.assertTrue(res_scipy.is_feasible)
            self.assertTrue(res_ortools.is_feasible)

            # Проверяем равенство суммарной стоимости с точностью до 4 знаков
            # (небольшое расхождение может быть из-за масштабирования float -> int64 в OR-Tools)
            self.assertAlmostEqual(
                res_scipy.total_cost,
                res_ortools.total_cost,
                places=3,
                msg=f"Стоимости расходятся на размере N={n}: SciPy={res_scipy.total_cost}, OR-Tools={res_ortools.total_cost}",
            )

    def test_batch_solving(self):
        gen = Spatial2DGenerator(metric="euclidean", seed=123)
        batch = gen.generate_batch(batch_size=5, n=20)

        results_scipy = self.scipy_solver.solve_batch(batch)
        results_ortools = self.ortools_solver.solve_batch(batch)

        self.assertEqual(len(results_scipy), 5)
        self.assertEqual(len(results_ortools), 5)

        for i in range(5):
            self.assertTrue(results_scipy[i].is_feasible)
            self.assertTrue(results_ortools[i].is_feasible)
            self.assertAlmostEqual(
                results_scipy[i].total_cost,
                results_ortools[i].total_cost,
                places=3,
            )


if __name__ == "__main__":
    unittest.main()
