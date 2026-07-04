import { pipeline, type ImageFeatureExtractionPipeline } from '@huggingface/transformers'
import { hashFile, getCachedEmbeddings, putEmbeddings } from './embeddingCache.ts'

export const MODEL_ID = 'Xenova/clip-vit-base-patch32'

export type Device = 'webgpu' | 'wasm'

/** The device actually selected for inference, once known. Read by the UI to show a GPU/CPU indicator. */
let resolvedDevice: Device | null = null
export function getResolvedDevice() {
  return resolvedDevice
}

/**
 * `navigator.gpu` existing doesn't guarantee a GPU adapter is actually available
 * (e.g. disabled drivers, sandboxed/headless environments), so we probe for a
 * real adapter rather than trusting feature detection alone.
 */
async function detectDevice(): Promise<Device> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return 'wasm'
  try {
    const adapter = await (navigator as any).gpu.requestAdapter()
    return adapter ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}

let extractorPromise: Promise<ImageFeatureExtractionPipeline> | null = null

/** Lazily loads (and caches) the CLIP image encoder. */
export function getImageEmbedder(onProgress?: (status: string) => void) {
  if (!extractorPromise) {
    extractorPromise = detectDevice().then(async (device) => {
      try {
        const embedder = await pipeline('image-feature-extraction', MODEL_ID, {
          device,
          dtype: device === 'webgpu' ? 'fp32' : 'q8',
          progress_callback: (progress: any) => {
            if (progress.status === 'progress') {
              onProgress?.(`${progress.file} — ${Math.round(progress.progress)}%`)
            } else {
              onProgress?.(progress.status)
            }
          },
        })
        resolvedDevice = device
        return embedder
      } catch (error) {
        if (device === 'wasm') throw error
        console.error('WebGPU pipeline failed to load, falling back to WASM:', error)
        resolvedDevice = 'wasm'
        return pipeline('image-feature-extraction', MODEL_ID, { device: 'wasm', dtype: 'q8' })
      }
    })
  }
  return extractorPromise
}

/** Runs an image through CLIP and returns its embedding as a plain number array. */
export async function embedImage(image: string | File, onProgress?: (status: string) => void) {
  const embedder = await getImageEmbedder(onProgress)
  const source = image instanceof File ? URL.createObjectURL(image) : image
  try {
    const output = await embedder(source)
    return Array.from(output.data as Float32Array)
  } finally {
    if (image instanceof File) URL.revokeObjectURL(source)
  }
}

export type EmbeddedImage = {
  name: string
  embedding: number[]
  file: File
}

/**
 * Embeds a batch of image files sequentially (the model runs on a single
 * WASM/WebGPU context, so parallelizing calls doesn't help and just spikes
 * memory). Files that fail to embed are skipped rather than aborting the batch.
 *
 * Files whose embeddings are already in the IndexedDB cache are returned
 * immediately without running the model. The `onProgress` 4th arg signals
 * whether the current item was served from cache.
 */
export async function embedImages(
  files: File[],
  onProgress?: (done: number, total: number, name: string, fromCache: boolean) => void,
): Promise<EmbeddedImage[]> {
  // Hash all files in parallel (cheap compared to inference).
  const hashes = await Promise.all(files.map(hashFile))
  const ids = hashes.map((h) => `${MODEL_ID}:${h}`)

  const cached = await getCachedEmbeddings(ids)

  // Only load the model if there are cache misses.
  const hasMisses = ids.some((id) => !cached.has(id))
  const embedder = hasMisses ? await getImageEmbedder() : null

  const results: EmbeddedImage[] = []
  const toStore: { id: string; embedding: number[] }[] = []

  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    const id = ids[i]
    const hit = cached.get(id)

    if (hit) {
      onProgress?.(i, files.length, file.name, true)
      results.push({ name: file.name, embedding: hit, file })
      continue
    }

    onProgress?.(i, files.length, file.name, false)
    const source = URL.createObjectURL(file)
    try {
      const output = await embedder!(source)
      const embedding = Array.from(output.data as Float32Array)
      results.push({ name: file.name, embedding, file })
      toStore.push({ id, embedding })
    } catch (error) {
      console.error(`Failed to embed ${file.name}:`, error)
    } finally {
      URL.revokeObjectURL(source)
    }
  }

  onProgress?.(files.length, files.length, '', false)
  await putEmbeddings(toStore)
  return results
}
