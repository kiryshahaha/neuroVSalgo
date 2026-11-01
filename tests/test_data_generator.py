import unittest
import numpy as np

from src.data_generator import (
    UniformGenerator,
    LognormalGenerator,
    Spatial2DGenerator,
    get_generator,
)


class TestDataGenerators(unittest.TestCase):
    def setUp(self):
        self.sizes = [10, 25, 50]
        self.batch_size = 4

    def test_uniform_generator_shapes_and_bounds(self):
        low, high = 0.0, 1.0
        gen = UniformGenerator(low=low, high=high, seed=42)

        for n in self.sizes:
            matrix = gen.generate(n)
            self.assertEqual(matrix.shape, (n, n))
            self.assertTrue(np.all(matrix >= low))
            self.assertTrue(np.all(matrix <= high))

        batch = gen.generate_batch(self.batch_size, n=15)
        self.assertEqual(batch.shape, (self.batch_size, 15, 15))
        self.assertTrue(np.all(batch >= low))
        self.assertTrue(np.all(batch <= high))

    def test_lognormal_generator_shapes_and_values(self):
        gen = LognormalGenerator(mean=0.0, sigma=0.5, normalize_max=True, seed=42)

        for n in self.sizes:
            matrix = gen.generate(n)
            self.assertEqual(matrix.shape, (n, n))
            self.assertTrue(np.all(matrix > 0))
            self.assertAlmostEqual(float(np.max(matrix)), 1.0, places=5)

        batch = gen.generate_batch(self.batch_size, n=15)
        self.assertEqual(batch.shape, (self.batch_size, 15, 15))
        for i in range(self.batch_size):
            self.assertAlmostEqual(float(np.max(batch[i])), 1.0, places=5)

    def test_spatial_generator_shapes_and_properties(self):
        gen = Spatial2DGenerator(metric="euclidean", seed=42)

        for n in self.sizes:
            matrix, workers, tasks = gen.generate_with_coordinates(n)
            self.assertEqual(matrix.shape, (n, n))
            self.assertEqual(workers.shape, (n, 2))
            self.assertEqual(tasks.shape, (n, 2))
            self.assertTrue(np.all(matrix >= 0))

        batch = gen.generate_batch(self.batch_size, n=15)
        self.assertEqual(batch.shape, (self.batch_size, 15, 15))
        self.assertTrue(np.all(batch >= 0))

    def test_reproducibility_with_seed(self):
        gen1 = UniformGenerator(seed=123)
        gen2 = UniformGenerator(seed=123)
        m1 = gen1.generate(20)
        m2 = gen2.generate(20)
        np.testing.assert_array_equal(m1, m2)

    def test_factory_get_generator(self):
        gen_u = get_generator("uniform", seed=1)
        self.assertIsInstance(gen_u, UniformGenerator)

        gen_l = get_generator("lognormal", seed=2)
        self.assertIsInstance(gen_l, LognormalGenerator)

        gen_s = get_generator("spatial_2d", seed=3)
        self.assertIsInstance(gen_s, Spatial2DGenerator)

        with self.assertRaises(ValueError):
            get_generator("unknown_distribution")


if __name__ == "__main__":
    unittest.main()
