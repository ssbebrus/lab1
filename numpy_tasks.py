import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return (np.asarray(data.matrix) > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    return [np.unique(row).tolist() for row in np.asarray(data.matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    return [np.unique(column).tolist() for column in np.asarray(data.matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(data.mean, data.std, (data.rows, data.columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = np.indices((data.rows, data.columns))
    return np.where((rows + columns) % 2 == 0, data.first, data.second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    left = (data.image_width - data.width) // 2
    top = (data.image_height - data.height) // 2
    image[max(top, 0):min(top + data.height, data.image_height),
          max(left, 0):min(left + data.width, data.image_width)] = data.shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color
    y, x = np.ogrid[:data.image_height, :data.image_width]
    center_x = (data.image_width - 1) / 2
    center_y = (data.image_height - 1) / 2
    mask = ((x - center_x) / data.semi_axis_x) ** 2 + ((y - center_y) / data.semi_axis_y) ** 2 <= 1
    image[mask] = data.shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    if not 1 <= data.window <= len(values):
        raise ValueError('Window must be between 1 and the series length')
    middle = values[1:-1]
    maxima = np.flatnonzero((middle > values[:-2]) & (middle > values[2:])) + 1
    minima = np.flatnonzero((middle < values[:-2]) & (middle < values[2:])) + 1
    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=np.convolve(values, np.ones(data.window) / data.window, mode='valid'),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if class_count < 0 or np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError('Class labels are out of range')
    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
