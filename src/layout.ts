import { UMAP } from 'umap-js'

function l2Normalize(v: number[]): number[] {
  let sum = 0
  for (const x of v) sum += x * x
  const inv = sum === 0 ? 0 : 1 / Math.sqrt(sum)
  return v.map((x) => x * inv)
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
): Promise<[number, number][]> {
  const n = embeddings.length

  if (n === 0) return []

  const data = embeddings.map(l2Normalize)

  // UMAP requires n > nNeighbors; for tiny inputs fall back to PCA-like spread.
  if (n <= 3) {
    return data.map((_, i) => [(i % 2) / Math.max(n - 1, 1), Math.floor(i / 2) / Math.max(n - 1, 1)] as [number, number])
  }

  const nNeighbors = Math.min(15, n - 1)
  const umap = new UMAP({ nComponents: 2, nNeighbors, minDist: 0.1 })

  let totalEpochs = 0
  const projected = await umap.fitAsync(data, (epoch) => {
    totalEpochs = epoch
    onProgress?.(epoch, 0)
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
