"""Simple randomized maze generator.

This module implements a readable, educational DFS-based maze generator
that outputs a binary occupancy grid suitable for the SpheroEnv `occupancy_grid`
argument. The grid uses 1 for walls/obstacles and 0 for free cells.

Key design choices for readability:
- Straightforward recursive backtracker / iterative DFS implementation
- Small, well-named functions and docstrings
- Deterministic behaviour via optional `seed` parameter
"""
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np


def _neighbors(cell: Tuple[int, int], width: int, height: int):
    """Get the neighboring cells of a given cell within bounds."""
    x, y = cell
    nbrs = []
    if x > 0:
        nbrs.append((x - 1, y))
    if x < width - 1:
        nbrs.append((x + 1, y))
    if y > 0:
        nbrs.append((x, y - 1))
    if y < height - 1:
        nbrs.append((x, y + 1))
    return nbrs


class MazeGenerator:
    """Generate a toy maze by carving a path on a 5x5 grid.

    Maze cells are returned as a binary occupancy grid where:
      - 1 indicates an occupied cell (wall)
      - 0 indicates free cell (corridor)

    A simple path is carved on a fixed 5x5 logical grid, then that grid is
    upscaled to an exact NxN occupancy grid.
    """

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        seed: Optional[int] = None,
    ):
        """Initialize the maze generator.

        Args:
            width: Target output width in occupancy cells.
            height: Target output height in occupancy cells.
            seed: Random seed for reproducibility.
        """
        self.width = int(width)
        self.height = int(height)
        self.seed = seed

    @staticmethod
    def _cell_center(index: int, target_size: int) -> int:
        """Map a logical 5x5 index to the center occupancy index in target size."""
        interior = max(1, target_size - 2)
        center = 1 + int(np.floor((index + 0.5) * interior / 5.0))
        return int(np.clip(center, 0, target_size - 1))

    def generate(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Return (occupancy_grid, start_cell, goal_cell).

        start_cell and goal_cell are logical (x, y) indices on the 5x5 toy grid.
        """
        rng = random.Random(self.seed)
        start_cell = (2, 2)

        carved = {start_cell}
        frontier = [start_cell]
        target_open_cells = 24
        connections: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        while frontier and len(carved) < target_open_cells:
            base = frontier[-1] if rng.random() < 0.35 else rng.choice(frontier)
            uncarved_neighbors = [cell for cell in _neighbors(base, 5, 5) if cell not in carved]
            if not uncarved_neighbors:
                frontier = [cell for cell in frontier if any(n not in carved for n in _neighbors(cell, 5, 5))]
                continue

            nxt = rng.choice(uncarved_neighbors)
            carved.add(nxt)
            frontier.append(nxt)
            connections.append((base, nxt))

        # Choose the farthest carved cell as the goal so the path spans the toy maze.
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        goal_cell = max(carved, key=lambda cell: (manhattan(cell, start_cell), cell[1], cell[0]))

        target_n = max(5, max(self.width, self.height))
        grid = np.ones((target_n, target_n), dtype=np.uint8)
        corridor_half_width = max(0, target_n // 40)

        def carve_square(row: int, col: int):
            r0 = max(1, row - corridor_half_width)
            r1 = min(target_n - 2, row + corridor_half_width)
            c0 = max(1, col - corridor_half_width)
            c1 = min(target_n - 2, col + corridor_half_width)
            grid[r0:r1 + 1, c0:c1 + 1] = 0

        def carve_corridor(a: Tuple[int, int], b: Tuple[int, int]):
            ar = self._cell_center(a[1], target_n)
            ac = self._cell_center(a[0], target_n)
            br = self._cell_center(b[1], target_n)
            bc = self._cell_center(b[0], target_n)

            carve_square(ar, ac)
            if ac != bc:
                c0, c1 = sorted((ac, bc))
                grid[max(1, ar - corridor_half_width):min(target_n - 1, ar + corridor_half_width + 1), c0:c1 + 1] = 0
            if ar != br:
                r0, r1 = sorted((ar, br))
                grid[r0:r1 + 1, max(1, bc - corridor_half_width):min(target_n - 1, bc + corridor_half_width + 1)] = 0
            carve_square(br, bc)

        for cell in carved:
            carve_square(self._cell_center(cell[1], target_n), self._cell_center(cell[0], target_n))

        for base, nxt in connections:
            carve_corridor(base, nxt)

        return grid, start_cell, goal_cell


def grid_to_world(
    cell: Tuple[int, int],
    maze_width: int,
    maze_height: int,
    cell_size: float,
) -> Tuple[float, float]:
    """Convert a logical toy-grid cell (x,y) to world coordinates (meters)."""
    x_cell, y_cell = cell
    target_n = max(5, max(int(maze_width), int(maze_height)))
    j = MazeGenerator._cell_center(int(x_cell), target_n)
    i = MazeGenerator._cell_center(int(y_cell), target_n)

    # cell centre relative to top-left of occupancy grid (meters)
    cx = (j + 0.5) * cell_size
    cy = (i + 0.5) * cell_size

    # convert to world coords with origin at occupancy-grid center
    world_w = target_n * cell_size
    world_h = target_n * cell_size
    world_x = cx - world_w / 2.0
    world_y = (world_h / 2.0) - cy  # invert y so that increasing cell y goes up
    return float(world_x), float(world_y)


if __name__ == "__main__":
    # Small demo when run as script
    mg = MazeGenerator(width=11, height=9, seed=0)
    grid, s, g = mg.generate()
    print("Start cell:", s)
    print("Goal cell:", g)
    print(grid.astype(int))
