import numpy as np
import pytest

from data.custom_datasets import MazeImageFolder

# Maze pixel encoding (float32 images in [0, 1])
WALL = [0, 0, 0]
CORRIDOR = [0, 0, 1]
START = [1, 0, 0]
END = [0, 1, 0]

ROUTE_LENGTH = 10

# Direction label encoding used by the solver/loss
UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


def _solver(route_length=ROUTE_LENGTH):
    """MazeImageFolder without running __init__ (get_solution only needs maze_route_length)."""
    ds = MazeImageFolder.__new__(MazeImageFolder)
    ds.maze_route_length = route_length
    return ds


def _maze(rows):
    """Build an (H, W, 3) float maze image from a grid of pixel codes."""
    return np.array(rows, dtype=np.float32)


def _replays_to_end(maze, path):
    """Walk the direction labels from the start marker.

    Returns True iff every move lands on a traversable in-bounds cell and the
    final position is the end marker (WAIT leaves the position unchanged).
    """
    y, x = np.argwhere((maze == START).all(axis=2))[0]
    end_y, end_x = np.argwhere((maze == END).all(axis=2))[0]
    moves = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
    for direction in path:
        if direction == WAIT:
            continue
        dy, dx = moves[int(direction)]
        y, x = y + dy, x + dx
        if not (0 <= y < maze.shape[0] and 0 <= x < maze.shape[1]):
            return False
        cell = maze[y, x]
        if not ((cell == CORRIDOR).all() or (cell == END).all() or (cell == START).all()):
            return False
    return (y, x) == (end_y, end_x)


def test_greedy_path_unchanged_on_simple_maze():
    """A maze the original greedy walk can solve must keep identical labels."""
    maze = _maze([
        [WALL, WALL, WALL, WALL, WALL],
        [WALL, START, CORRIDOR, END, WALL],
        [WALL, WALL, WALL, WALL, WALL],
    ])
    path = _solver().get_solution(maze)
    expected = np.array([RIGHT, RIGHT] + [WAIT] * (ROUTE_LENGTH - 2))
    np.testing.assert_array_equal(path, expected)


def test_dead_end_maze_falls_back_to_bfs():
    """Greedy walk gets stuck in a dead end; BFS must still produce a valid route.

    Layout: from START, the highest-priority direction (Up) is a dead-end cell.
    The real route goes right along row 2, down column 3, then left to END.
    """
    maze = _maze([
        [WALL, WALL, WALL, WALL, WALL],
        [WALL, CORRIDOR, WALL, CORRIDOR, WALL],
        [WALL, START, CORRIDOR, CORRIDOR, WALL],
        [WALL, WALL, WALL, CORRIDOR, WALL],
        [WALL, END, CORRIDOR, CORRIDOR, WALL],
        [WALL, WALL, WALL, WALL, WALL],
    ])
    path = _solver().get_solution(maze)
    assert set(np.unique(path)) <= {UP, DOWN, LEFT, RIGHT, WAIT}
    assert -1 not in path
    assert _replays_to_end(maze, path)
    # BFS returns the shortest route: R, R, D, D, L, L
    np.testing.assert_array_equal(
        path, np.array([RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT] + [WAIT] * 4))


def test_unreachable_end_raises():
    maze = _maze([
        [WALL, WALL, WALL, WALL, WALL],
        [WALL, START, WALL, END, WALL],
        [WALL, WALL, WALL, WALL, WALL],
    ])
    with pytest.raises(ValueError, match="no valid path"):
        _solver().get_solution(maze)


def test_route_longer_than_budget_truncates():
    """Route (6 moves) longer than maze_route_length is sliced, not an error.

    Truncation mid-route is the documented behaviour: sequences are sliced to
    maze_route_length. The fallback path is the shortest route's prefix.
    """
    maze = _maze([
        [WALL, WALL, WALL, WALL, WALL],
        [WALL, CORRIDOR, WALL, CORRIDOR, WALL],
        [WALL, START, CORRIDOR, CORRIDOR, WALL],
        [WALL, WALL, WALL, CORRIDOR, WALL],
        [WALL, END, CORRIDOR, CORRIDOR, WALL],
        [WALL, WALL, WALL, WALL, WALL],
    ])
    path = _solver(route_length=5).get_solution(maze)
    np.testing.assert_array_equal(
        path, np.array([RIGHT, RIGHT, DOWN, DOWN, LEFT]))
    assert len(path) == 5


@pytest.mark.parametrize("missing", ["start", "end"])
def test_missing_markers_raise(missing):
    rows = [
        [WALL, WALL, WALL, WALL, WALL],
        [WALL, CORRIDOR, CORRIDOR, CORRIDOR, WALL],
        [WALL, WALL, WALL, WALL, WALL],
    ]
    rows[1][1] = START if missing == "end" else END
    with pytest.raises(ValueError, match="marker"):
        _solver().get_solution(_maze(rows))
