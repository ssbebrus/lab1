import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import python_basics as basic
import numpy_tasks as tasks
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)


class PythonTests(unittest.TestCase):
    def test_vowels(self):
        for text, expected in [('', 0), ('AEiou', 5), ('Python', 1), ('xyz', 0)]:
            with self.subTest(text=text):
                self.assertEqual(basic.count_vowels(TextInput(text)), expected)

    def test_unique(self):
        for text, expected in [('', True), ('abc', True), ('aba', False), ('aA', True)]:
            with self.subTest(text=text):
                self.assertEqual(basic.has_unique_characters(TextInput(text)), expected)

    def test_bits(self):
        for number, expected in [(1, 1), (8, 1), (15, 4), (123, 6)]:
            with self.subTest(number=number):
                self.assertEqual(basic.count_one_bits(PositiveIntegerInput(number)), expected)

    def test_persistence(self):
        for number, expected in [(39, 3), (4, 0), (999, 4), (10, 1), (1, 0)]:
            with self.subTest(number=number):
                self.assertEqual(basic.multiplicative_persistence(PositiveIntegerInput(number)), expected)

    def test_mse(self):
        for x, y, expected in [([1], [1], 0), ([1, 2], [2, 4], 2.5), ([-1, 3], [1, -1], 10)]:
            with self.subTest(x=x, y=y):
                self.assertEqual(basic.mse(VectorPairInput(x, y)), expected)

    def test_factorization(self):
        for number, expected in [(1, ''), (2, '(2)'), (49, '(7**2)'), (86240, '(2**5)(5)(7**2)(11)')]:
            with self.subTest(number=number):
                self.assertEqual(basic.prime_factorization(PositiveIntegerInput(number)), expected)

    def test_pyramid(self):
        for number, expected in [(1, 1), (5, 2), (14, 3), (30, 4), (2, 'It is impossible')]:
            with self.subTest(number=number):
                self.assertEqual(basic.pyramid(PositiveIntegerInput(number)), expected)

    def test_balanced(self):
        for number, expected in [(1, True), (12, True), (123, False), (1234006, True), (123456, False), (1221, True)]:
            with self.subTest(number=number):
                self.assertEqual(basic.is_balanced_number(PositiveIntegerInput(number)), expected)


class NumpyTests(unittest.TestCase):
    def test_sum_prod(self):
        for size in [1, 2, 4]:
            with self.subTest(size=size):
                matrices = np.array([np.eye(size), np.eye(size) * 2])
                vectors = np.arange(2 * size).reshape(2, size, 1)
                assert_allclose(tasks.sum_prod(MatrixVectorBatchInput(matrices, vectors)), vectors[0] + 2 * vectors[1])
        assert_array_equal(tasks.sum_prod(MatrixVectorBatchInput([[[2]]], [[[3]]])), [[6]])

    def test_binarize(self):
        matrix = np.array([[0, 0.5, 1], [-1, 2, 0.5]])
        original = matrix.copy()
        assert_array_equal(tasks.binarize(BinarizeInput(matrix)), [[0, 0, 1], [0, 1, 0]])
        assert_array_equal(matrix, original)
        assert_array_equal(tasks.binarize(BinarizeInput([[1, 2]], 2)), [[0, 0]])
        self.assertEqual(tasks.binarize(BinarizeInput(np.empty((0, 3)))).shape, (0, 3))

    def test_unique_rows(self):
        self.assertEqual(tasks.unique_rows(MatrixInput([[2, 1, 2], [3, 3, 3]])), [[1, 2], [3]])
        self.assertEqual(tasks.unique_rows(MatrixInput([[1]])), [[1]])
        self.assertEqual(tasks.unique_rows(MatrixInput(np.empty((0, 2)))), [])

    def test_unique_columns(self):
        self.assertEqual(tasks.unique_columns(MatrixInput([[2, 1, 2], [3, 1, 3]])), [[2, 3], [1], [2, 3]])
        self.assertEqual(tasks.unique_columns(MatrixInput([[1]])), [[1]])
        self.assertEqual(tasks.unique_columns(MatrixInput(np.empty((2, 0)))), [])

    def test_statistics(self):
        for rows, columns in [(1, 1), (3, 4)]:
            with self.subTest(shape=(rows, columns)):
                data = RandomMatrixInput(rows, columns, 2, 3, 17)
                result = tasks.matrix_statistics(data)
                self.assertEqual(result.matrix.shape, (rows, columns))
                assert_array_equal(result.matrix, tasks.matrix_statistics(data).matrix)
                assert_allclose(result.row_means, [sum(row) / columns for row in result.matrix])
                assert_allclose(result.column_means, [sum(col) / rows for col in result.matrix.T])
                assert_allclose(result.row_variances, [sum((row - row.mean()) ** 2) / columns for row in result.matrix])
                assert_allclose(result.column_variances, [sum((col - col.mean()) ** 2) / rows for col in result.matrix.T])
        result = tasks.matrix_statistics(RandomMatrixInput(2, 3, 7, 0))
        assert_array_equal(result.matrix, np.full((2, 3), 7))
        assert_array_equal(result.row_variances, [0, 0])

    def test_chess(self):
        assert_array_equal(tasks.chess(ChessInput(3, 4, 0, 1)), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        assert_array_equal(tasks.chess(ChessInput(1, 1, 5, 8)), [[5]])
        assert_array_equal(tasks.chess(ChessInput(2, 1, 1.5, 2.5)), [[1.5], [2.5]])

    def test_rectangle(self):
        for height, width in [(3, 5), (4, 6), (1, 1)]:
            with self.subTest(size=(height, width)):
                result = tasks.draw_rectangle(RectangleInput(width, height, height + 4, width + 4, (255, 0, 0), (0, 0, 0)))
                self.assertEqual(result.shape, (height + 4, width + 4, 3))
                self.assertEqual(np.count_nonzero(result[:, :, 0]), height * width)
                assert_array_equal(result[2:2 + height, 2:2 + width, 0], np.full((height, width), 255))
                assert_array_equal(result[0, 0], [0, 0, 0])
        assert_array_equal(tasks.draw_rectangle(RectangleInput(8, 8, 2, 2, (1, 2, 3), (0, 0, 0))), np.tile([1, 2, 3], (2, 2, 1)))

    def test_ellipse(self):
        for height, width, axis_x, axis_y in [(5, 7, 2, 1), (4, 6, 2, 1), (1, 1, 1, 1)]:
            with self.subTest(size=(height, width)):
                result = tasks.draw_ellipse(EllipseInput(axis_x, axis_y, height, width, (255, 20, 30), (1, 2, 3)))
                self.assertEqual(result.shape, (height, width, 3))
                assert_array_equal(result, result[::-1])
                assert_array_equal(result, result[:, ::-1])
                for y in range(height):
                    for x in range(width):
                        inside = ((x - (width - 1) / 2) / axis_x) ** 2 + ((y - (height - 1) / 2) / axis_y) ** 2 <= 1
                        assert_array_equal(result[y, x], [255, 20, 30] if inside else [1, 2, 3])

    def test_time_series(self):
        result = tasks.analyze_time_series(TimeSeriesInput([1, 3, 1, 0, 2], 2))
        assert_array_equal(result.local_maxima_indices, [1])
        assert_array_equal(result.local_minima_indices, [3])
        assert_allclose(result.moving_average, [2, 2, 0.5, 1])
        self.assertAlmostEqual(result.mean, 1.4)
        self.assertAlmostEqual(result.variance, 1.04)
        self.assertAlmostEqual(result.std, 1.04 ** 0.5)
        for values, window, expected in [([5], 1, [5]), ([1, 1, 1], 3, [1]), ([1, 2], 1, [1, 2])]:
            with self.subTest(values=values, window=window):
                result = tasks.analyze_time_series(TimeSeriesInput(values, window))
                self.assertEqual(len(result.local_maxima_indices), 0)
                self.assertEqual(len(result.local_minima_indices), 0)
                assert_allclose(result.moving_average, expected)
        for window in [0, 4]:
            with self.assertRaises(ValueError):
                tasks.analyze_time_series(TimeSeriesInput([1, 2, 3], window))

    def test_one_hot(self):
        assert_array_equal(tasks.one_hot(OneHotInput([0, 2, 3, 0])), [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        assert_array_equal(tasks.one_hot(OneHotInput([1], 4)), [[0, 1, 0, 0]])
        self.assertEqual(tasks.one_hot(OneHotInput([])).shape, (0, 0))
        self.assertEqual(tasks.one_hot(OneHotInput([], 3)).shape, (0, 3))
        for labels, count in [([-1], 2), ([2], 2), ([], -1)]:
            with self.assertRaises(ValueError):
                tasks.one_hot(OneHotInput(labels, count))


if __name__ == '__main__':
    unittest.main()
