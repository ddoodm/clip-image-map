/**
 * Tiny seeded PRNG (mulberry32) so a UMAP run can be reproduced from a seed,
 * or a fresh one can be rolled on demand. `umap-js` just needs a `() =>
 * number` in [0, 1), same shape as `Math.random`.
 */
export function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A random 32-bit seed suitable for `mulberry32`. */
export function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0
}
