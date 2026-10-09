from __future__ import annotations


def supercover_line_cells(
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return every raster cell touched by the segment from start to goal."""
    row0, col0 = start
    row1, col1 = goal
    delta_row = row1 - row0
    delta_col = col1 - col0
    row_steps = abs(delta_row)
    col_steps = abs(delta_col)
    row_sign = 1 if delta_row > 0 else -1
    col_sign = 1 if delta_col > 0 else -1
    row = row0
    col = col0
    row_index = 0
    col_index = 0
    cells = [(row, col)]

    while row_index < row_steps or col_index < col_steps:
        row_crossing = (1 + 2 * row_index) * col_steps
        col_crossing = (1 + 2 * col_index) * row_steps
        if row_crossing == col_crossing:
            next_row = row + row_sign
            next_col = col + col_sign
            cells.extend(((next_row, col), (row, next_col), (next_row, next_col)))
            row = next_row
            col = next_col
            row_index += 1
            col_index += 1
        elif row_crossing < col_crossing:
            row += row_sign
            row_index += 1
            cells.append((row, col))
        else:
            col += col_sign
            col_index += 1
            cells.append((row, col))

    return list(dict.fromkeys(cells))
