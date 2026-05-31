"""Stub: 9x9 boards are produced by `samplers.Sudoku9x9.sudoku9x9_sampler.Sampler`.
The non-tensorized `get_mnist_sudoku9x9_dataset` path that depends on
`generate_boards_set` is not used in the GPU training pipeline; the
tensorized path goes through the sampler instead."""


def generate_boards_set():
    raise NotImplementedError(
        "9x9 board generation is handled by samplers.Sudoku9x9.sudoku9x9_sampler.Sampler; "
        "call `tensorized_get_mnist_sudoku9x9_dataset` instead of `get_mnist_sudoku9x9_dataset`."
    )
