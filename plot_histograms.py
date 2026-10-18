from pathlib import Path

import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(matrix):
    row_figure, row_axes = plt.subplots(matrix.shape[0], 1, squeeze=False,
                                      figsize=(7, 2.5 * matrix.shape[0]))
    for index, row in enumerate(matrix):
        row_axes[index, 0].hist(row, bins=10)
        row_axes[index, 0].set_title(f'Row {index + 1}')
    row_figure.tight_layout()

    column_figure, column_axes = plt.subplots(matrix.shape[1], 1, squeeze=False,
                                            figsize=(7, 2.5 * matrix.shape[1]))
    for index, column in enumerate(matrix.T):
        column_axes[index, 0].hist(column, bins=10)
        column_axes[index, 0].set_title(f'Column {index + 1}')
    column_figure.tight_layout()
    return row_figure, column_figure


if __name__ == '__main__':
    statistics = matrix_statistics(RandomMatrixInput(4, 5))
    rows, columns = plot_histograms(statistics.matrix)
    output = Path(__file__).parent / 'plots'
    output.mkdir(exist_ok=True)
    rows.savefig(output / 'rows.png')
    columns.savefig(output / 'columns.png')
    plt.close('all')
