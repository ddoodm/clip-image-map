const DB_NAME = 'clip-image-map'
const STORE_NAME = 'embeddings'
const DB_VERSION = 1

type CacheRecord = {
  id: string
  embedding: number[]
}

let dbPromise: Promise<IDBDatabase> | null = null

function openDB(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = () => {
        req.result.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
  }
  return dbPromise
}

/** SHA-256 hex digest of the file's raw bytes. */
export async function hashFile(file: File): Promise<string> {
  const buf = await file.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

/**
 * Batch-read embeddings from IndexedDB.
 * Returns a Map of id -> embedding for each id that was found.
 * Silently returns an empty Map if IndexedDB is unavailable.
 */
export async function getCachedEmbeddings(ids: string[]): Promise<Map<string, number[]>> {
  const result = new Map<string, number[]>()
  if (ids.length === 0) return result
  try {
    const db = await openDB()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly')
      const store = tx.objectStore(STORE_NAME)
      let pending = ids.length
      for (const id of ids) {
        const req = store.get(id)
        req.onsuccess = () => {
          const record = req.result as CacheRecord | undefined
          if (record) result.set(record.id, record.embedding)
          if (--pending === 0) resolve()
        }
        req.onerror = () => {
          if (--pending === 0) resolve()
        }
      }
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('[embeddingCache] read failed, treating all as cache misses:', err)
  }
  return result
}

/**
 * Batch-write embeddings to IndexedDB.
 * Silently swallows errors so a write failure never breaks embedding.
 */
export async function putEmbeddings(records: CacheRecord[]): Promise<void> {
  if (records.length === 0) return
  try {
    const db = await openDB()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite')
      const store = tx.objectStore(STORE_NAME)
      for (const record of records) {
        store.put(record)
      }
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('[embeddingCache] write failed:', err)
  }
}
