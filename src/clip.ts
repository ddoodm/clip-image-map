import { AutoProcessor, SiglipVisionModel, RawImage } from '@huggingface/transformers'
import { hashFile, getCachedEmbeddings, putEmbeddings } from './embeddingCache.ts'

export const MODEL_ID = 'onnx-community/siglip2-base-patch16-224-ONNX'

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

type Embedder = {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>
  model: Awaited<ReturnType<typeof SiglipVisionModel.from_pretrained>>
}

let embedderPromise: Promise<Embedder> | null = null

/**
 * Lazily loads (and caches) the SigLIP2 vision encoder and its image
 * processor. `SiglipVisionModel.from_pretrained` fetches only the
 * `vision_model*.onnx` weights (not the text tower), so this stays a
 * single-purpose image encoder despite SigLIP2 being a dual-encoder model.
 */
export function getImageEmbedder(onProgress?: (status: string) => void): Promise<Embedder> {
  if (!embedderPromise) {
    embedderPromise = (async () => {
      const processor = await AutoProcessor.from_pretrained(MODEL_ID)
      const device = await detectDevice()

      const progress_callback = (progress: any) => {
        if (progress.status === 'progress') {
          onProgress?.(`${progress.file} — ${Math.round(progress.progress)}%`)
        } else {
          onProgress?.(progress.status)
        }
      }

      try {
        const model = await SiglipVisionModel.from_pretrained(MODEL_ID, {
          device,
          dtype: device === 'webgpu' ? 'fp32' : 'q8',
          progress_callback,
        })
        resolvedDevice = device
        return { processor, model }
      } catch (error) {
        if (device === 'wasm') throw error
        console.error('WebGPU model failed to load, falling back to WASM:', error)
        resolvedDevice = 'wasm'
        const model = await SiglipVisionModel.from_pretrained(MODEL_ID, { device: 'wasm', dtype: 'q8' })
        return { processor, model }
      }
    })()
  }
  return embedderPromise
}

/** Runs a decoded image through the processor + vision encoder and returns its pooled embedding. */
async function embedRawImage({ processor, model }: Embedder, image: RawImage): Promise<number[]> {
  const inputs = await processor(image)
  const { pooler_output } = await model(inputs)
  return Array.from(pooler_output.data as Float32Array)
}

/** Runs an image through SigLIP2 and returns its embedding as a plain number array. */
export async function embedImage(image: string | File, onProgress?: (status: string) => void) {
  const embedder = await getImageEmbedder(onProgress)
  const source = image instanceof File ? URL.createObjectURL(image) : image
  try {
    const raw = await RawImage.read(source)
    return await embedRawImage(embedder, raw)
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
      const raw = await RawImage.read(source)
      const embedding = await embedRawImage(embedder!, raw)
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
