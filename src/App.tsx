import { useEffect, useRef, useState } from 'react'
import { embedImages, getImageEmbedder, getResolvedDevice, type Device } from './clip.ts'
import { projectTo2D } from './layout.ts'
import { assignGrid } from './grid.ts'
import { ImageMap, type ImageMapItem } from './ImageMap.tsx'

const IMAGE_TYPE_RE = /^image\//
const IGNORED_EXTENSION_RE = /\.arw$/i

type MapState = {
  items: ImageMapItem[]
  cols: number
  rows: number
}

export function App() {
  const [status, setStatus] = useState('Loading CLIP model…')
  const [ready, setReady] = useState(false)
  const [working, setWorking] = useState(false)
  const [mapState, setMapState] = useState<MapState | null>(null)
  const [device, setDevice] = useState<Device | null>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getImageEmbedder((progress) => setStatus(`Loading model — ${progress}`)).then(() => {
      setStatus('Model ready — choose a folder')
      setReady(true)
      setDevice(getResolvedDevice())
    })
  }, [])

  useEffect(() => {
    folderInputRef.current?.setAttribute('webkitdirectory', '')
  }, [])

  async function handleFolderChange(event: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).filter(
      (file) => IMAGE_TYPE_RE.test(file.type) && !IGNORED_EXTENSION_RE.test(file.name),
    )
    if (files.length === 0) {
      setStatus('No images found in that folder')
      return
    }

    setWorking(true)
    setMapState(null)

    // 1. Embed
    let cachedCount = 0
    const embedded = await embedImages(files, (done, total, name, fromCache) => {
      if (fromCache) cachedCount++
      const cacheNote = cachedCount > 0 ? ` (${cachedCount} cached)` : ''
      setStatus(`Embedding ${done}/${total}${cacheNote}${name ? ` — ${name}` : ''}`)
    })

    if (embedded.length === 0) {
      setStatus('No images could be embedded')
      setWorking(false)
      return
    }

    setStatus(`Projecting ${embedded.length} embeddings…`)

    // 2. UMAP → 2D
    const points = await projectTo2D(
      embedded.map((e) => e.embedding),
      (epoch) => setStatus(`Projecting — epoch ${epoch}…`),
    )

    setStatus(`Arranging grid…`)

    // 3. Grid assignment
    const aspect = window.innerWidth / window.innerHeight
    const { cells, cols, rows } = assignGrid(points, aspect)

    const items: ImageMapItem[] = cells.map((cell) => ({
      index: cell.index,
      name: embedded[cell.index].name,
      file: embedded[cell.index].file,
      col: cell.col,
      row: cell.row,
    }))

    setMapState({ items, cols, rows })
    setStatus(`${embedded.length} images — scroll to zoom, drag to pan`)
    setWorking(false)
  }

  const showMap = mapState !== null

  return (
    <>
      <header className={showMap ? 'header header--overlay' : 'header'}>
        <h1>CLIP Image Map</h1>
        {device && (
          <p className="device-badge">
            {device === 'webgpu' ? '⚡ WebGPU' : '🐢 WASM'}
          </p>
        )}
        <p className="status">{status}</p>
        <input
          ref={folderInputRef}
          type="file"
          multiple
          disabled={!ready || working}
          onChange={handleFolderChange}
        />
      </header>

      {showMap && (
        <div className="map-container">
          <ImageMap items={mapState.items} cols={mapState.cols} rows={mapState.rows} />
        </div>
      )}
    </>
  )
}
