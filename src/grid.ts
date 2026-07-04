export type GridCell = {
  index: number
  col: number
  row: number
}

/**
 * Extra grid capacity per axis, beyond the minimum needed to fit every
 * point. Total cells scale roughly with OVERSIZE^2. This slack is what lets
 * density vary across the grid — sparse regions of the projection end up
 * with genuine empty cells, and the overall border traces the shape of the
 * point cloud instead of a perfect rectangle.
 */
const OVERSIZE = 1.25

/**
 * Assigns each point to a unique cell in a grid by rounding its normalized
 * [0,1] position to the nearest cell, then resolving collisions by nudging
 * to the nearest still-free cell (expanding ring search).
 *
 * Unlike a rectangle-filling approach, this preserves the actual shape of
 * the point cloud: dense regions pack tightly, sparse regions leave gaps,
 * and the outer boundary is organic rather than a straight edge.
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

  const cols = Math.max(1, Math.ceil(Math.sqrt(n * aspect) * OVERSIZE))
  const rows = Math.max(1, Math.ceil(Math.sqrt(n / aspect) * OVERSIZE))

  const key = (col: number, row: number) => row * cols + col
  const occupied = new Set<number>()
  const cells = new Array<GridCell>(n)

  for (let i = 0; i < n; i++) {
    const [x, y] = points[i]
    // Clamp so points at the x=1/y=1 edge don't round one cell out of bounds.
    const desiredCol = Math.min(cols - 1, Math.floor(x * cols))
    const desiredRow = Math.min(rows - 1, Math.floor(y * rows))

    const cell = findNearestFreeCell(desiredCol, desiredRow, occupied, cols, rows, key)
    occupied.add(key(cell.col, cell.row))
    cells[i] = { index: i, col: cell.col, row: cell.row }
  }

  return { cells, cols, rows }
}

/**
 * Finds the unoccupied cell closest to (col, row) via an expanding
 * Chebyshev-ring search, breaking ties by true Euclidean distance. Points
 * are processed in input order, so earlier points in dense clusters claim
 * their ideal cell first and later ones settle for the nearest vacancy.
 */
function findNearestFreeCell(
  col: number,
  row: number,
  occupied: Set<number>,
  cols: number,
  rows: number,
  key: (col: number, row: number) => number,
): { col: number; row: number } {
  if (!occupied.has(key(col, row))) return { col, row }

  const maxRadius = Math.max(cols, rows)
  for (let radius = 1; radius <= maxRadius; radius++) {
    let best: { col: number; row: number; dist: number } | null = null

    for (let dc = -radius; dc <= radius; dc++) {
      for (let dr = -radius; dr <= radius; dr++) {
        // Only the outer edge of this radius — smaller radii were already checked.
        if (Math.max(Math.abs(dc), Math.abs(dr)) !== radius) continue
        const c = col + dc
        const r = row + dr
        if (c < 0 || c >= cols || r < 0 || r >= rows) continue
        if (occupied.has(key(c, r))) continue
        const dist = dc * dc + dr * dr
        if (!best || dist < best.dist) best = { col: c, row: r, dist }
      }
    }

    if (best) return best
  }

  // Shouldn't happen given OVERSIZE > 1, but fall back to a linear scan.
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!occupied.has(key(c, r))) return { col: c, row: r }
    }
  }
  throw new Error('Grid has no free cells left')
}
