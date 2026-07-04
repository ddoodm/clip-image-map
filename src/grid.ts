export type GridCell = {
  index: number
  col: number
  row: number
}

/**
 * Assigns each point to a unique cell in a cols×rows grid using recursive
 * bisection (RasterFairy-style). Points are sorted by position at each level
 * before splitting, which keeps nearby points in the 2D projection spatially
 * close in grid space.
 *
 * @param points  Normalized [x, y] coords in [0, 1], one per item.
 * @param aspect  Desired width/height ratio for the grid (default 1).
 * @returns       Array of GridCell in the same order as `points`.
 */
export function assignGrid(
  points: [number, number][],
  aspect = 1,
): { cells: GridCell[]; cols: number; rows: number } {
  const n = points.length
  if (n === 0) return { cells: [], cols: 0, rows: 0 }

  const cols = Math.max(1, Math.ceil(Math.sqrt(n * aspect)))
  const rows = Math.ceil(n / cols)

  const cells = new Array<GridCell>(n)

  // Indices of every input point; we'll recursively partition these.
  const indices = Array.from({ length: n }, (_, i) => i)

  function assign(idx: number[], x0: number, y0: number, w: number, h: number): void {
    if (idx.length === 0 || w === 0 || h === 0) return

    if (idx.length === 1) {
      cells[idx[0]] = { index: idx[0], col: x0, row: y0 }
      return
    }

    const totalCells = w * h
    if (totalCells === 1) {
      // More points than cells in this block — assign them all to the same cell
      // (shouldn't happen with correct sizing, but guard anyway)
      for (const i of idx) cells[i] = { index: i, col: x0, row: y0 }
      return
    }

    // Split along the longer dimension
    const splitHoriz = w >= h
    if (splitHoriz) {
      const leftCols = Math.floor(w / 2)
      const rightCols = w - leftCols
      const leftCells = leftCols * h
      // Sort by x so the left half gets the lower-x points
      idx.sort((a, b) => points[a][0] - points[b][0])
      const split = Math.min(leftCells, idx.length)
      assign(idx.slice(0, split), x0, y0, leftCols, h)
      assign(idx.slice(split), x0 + leftCols, y0, rightCols, h)
    } else {
      const topRows = Math.floor(h / 2)
      const botRows = h - topRows
      const topCells = w * topRows
      // Sort by y
      idx.sort((a, b) => points[a][1] - points[b][1])
      const split = Math.min(topCells, idx.length)
      assign(idx.slice(0, split), x0, y0, w, topRows)
      assign(idx.slice(split), x0, y0 + topRows, w, botRows)
    }
  }

  assign(indices, 0, 0, cols, rows)
  return { cells, cols, rows }
}
