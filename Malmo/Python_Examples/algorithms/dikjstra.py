import math

def get_path_dikjstra(start, end, layout):
    """
        Finds the shortest path from source to destination on the map. It used the grid observation as the graph.
        See example on the Tutorial.pdf file for knowing which index should be north, south, west and east.

        Args
            layout:   <list>  list of block types string representing the blocks on the map.
            source:     <int>   source block index.
            dest:       <int>   destination block index.

        Returns
            path_list:  <list>  block indexes representing a path from source (first element) to destination (last)
        """

    n = len(layout)
    dim = int(math.sqrt(n))

    def get_row(idx):
        return int(idx / dim)

    def get_col(idx):
        return idx % dim

    def to_index(row, col):
        return (row * dim) + col

    start = to_index(start[0], start[1])
    end = to_index(end[0], end[1])

    def get_neighbors(idx):
        row = get_row(idx)
        col = get_col(idx)

        nbs = []

        if idx < (n - 1) and get_row(idx + 1) == row:
            nbs.append(idx + 1)

        if idx > 0 and get_row(idx - 1) == row:
            nbs.append(idx - 1)

        if row > 0:
            nbs.append(to_index(row - 1, col))

        if row < get_row(n - 1):
            nbs.append(to_index(row + 1, col))

        nbs = [idx for idx in nbs if layout[idx] not in [0, 2]]

        return nbs

    dist = {idx: float("inf") for idx in range(n) if layout[idx] not in [0, 2]}
    prev = {idx: None for idx in range(n) if layout[idx] not in [0, 2]}

    dist[start] = 0

    Q = [idx for idx in range(n) if layout[idx] not in [0, 2]]

    while len(Q) > 0:
        u = min(Q, key=lambda idx: dist[idx])

        if dist[u] == float("inf"):
            break

        Q.remove(u)

        for neighbor in get_neighbors(u):
            alt = dist[u] + 1

            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = u

    block_path = []

    u = end

    while prev[u] is not None:
        block_path = [u] + block_path
        u = prev[u]

    block_path = [u] + block_path

    return block_path, dim
    # -------------------------------------