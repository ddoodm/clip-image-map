import { UMAP } from 'umap-js'

function l2Normalize(v: number[]): number[] {
  let sum = 0
  for (const x of v) sum += x * x
  const inv = sum === 0 ? 0 : 1 / Math.sqrt(sum)
  return v.map((x) => x * inv)
}

export type ProjectOptions = {
  /**
   * Number of neighbors UMAP considers per point. Higher values weigh more
   * global structure (better relative placement of dissimilar clusters) at
   * the cost of local precision and speed. Capped at n - 1.
   */
  nNeighbors?: number
  /**
   * Minimum distance between points in the 2D embedding. Lower values pack
   * near-duplicates almost on top of each other, which — since points are
   * later snapped to a grid — causes ties to be broken by input order rather
   * than similarity. A slightly higher value spreads tight clusters out
   * before grid-snapping, usually improving final placement.
   */
  minDist?: number
  /**
   * Optimization steps. umap-js defaults this to 500/400/300/200 depending
   * on dataset size, which under-converges larger libraries. Pin a fixed
   * value to keep convergence quality independent of n.
   */
  nEpochs?: number
  /** Seed for the PRNG so re-running on the same photos gives the same layout. */
  random?: () => number
}

const DEFAULT_OPTIONS: Required<Pick<ProjectOptions, 'nNeighbors' | 'minDist' | 'nEpochs'>> = {
  nNeighbors: 30,
  minDist: 0.25,
  nEpochs: 500,
}

/**
 * Projects high-dimensional embeddings to 2D via UMAP, then normalizes to [0,1].
 *
 * onProgress receives epoch count during UMAP optimization so the caller can
 * update the UI without blocking. Returns an array of [x, y] coords in the
 * same order as the input.
 */
export async function projectTo2D(
  embeddings: number[][],
  onProgress?: (epoch: number, total: number) => void,
  options?: ProjectOptions,
): Promise<[number, number][]> {
  const n = embeddings.length

  if (n === 0) return []

  const data = embeddings.map(l2Normalize)

  // UMAP requires n > nNeighbors; for tiny inputs fall back to PCA-like spread.
  if (n <= 3) {
    return data.map((_, i) => [(i % 2) / Math.max(n - 1, 1), Math.floor(i / 2) / Math.max(n - 1, 1)] as [number, number])
  }

  const nNeighbors = Math.min(options?.nNeighbors ?? DEFAULT_OPTIONS.nNeighbors, n - 1)
  const minDist = options?.minDist ?? DEFAULT_OPTIONS.minDist
  const nEpochs = options?.nEpochs ?? DEFAULT_OPTIONS.nEpochs
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors,
    minDist,
    nEpochs,
    ...(options?.random ? { random: options.random } : {}),
  })

  const totalEpochs = nEpochs
  const projected = await umap.fitAsync(data, (epoch) => {
    onProgress?.(epoch, totalEpochs)
  })
  onProgress?.(totalEpochs, totalEpochs)

  // Normalize to [0, 1] square
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const [x, y] of projected) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1

  return projected.map(([x, y]) => [(x - minX) / rangeX, (y - minY) / rangeY])
}
