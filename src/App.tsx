import { useEffect, useRef, useState } from 'react'
import { embedImages, getImageEmbedder, getResolvedDevice, type Device, type EmbeddedImage } from './clip.ts'
import { projectTo2D } from './layout.ts'
import { assignGrid } from './grid.ts'
import { ImageMap, type ImageMapItem } from './ImageMap.tsx'
import { mulberry32, randomSeed } from './random.ts'

const IMAGE_TYPE_RE = /^image\//
const IGNORED_EXTENSION_RE = /\.arw$/i

type MapState = {
  items: ImageMapItem[]
  cols: number
  rows: number
}

type LayoutSettings = {
  nNeighbors: number
  minDist: number
  nEpochs: number
  seed: number
}

const DEFAULT_SETTINGS: Omit<LayoutSettings, 'seed'> = {
  nNeighbors: 30,
  minDist: 0.25,
  nEpochs: 500,
}

export function App() {
  const [status, setStatus] = useState('Loading CLIP model…')
  const [ready, setReady] = useState(false)
  const [working, setWorking] = useState(false)
  const [mapState, setMapState] = useState<MapState | null>(null)
  const [device, setDevice] = useState<Device | null>(null)
  const [embedded, setEmbedded] = useState<EmbeddedImage[]>([])
  const [settings, setSettings] = useState<LayoutSettings>(() => ({ ...DEFAULT_SETTINGS, seed: randomSeed() }))
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

  /** Re-projects already-embedded images into the grid using `layoutSettings`. */
  async function runLayout(images: EmbeddedImage[], layoutSettings: LayoutSettings) {
    setWorking(true)
    setStatus(`Projecting ${images.length} embeddings…`)

    // UMAP → 2D
    const points = await projectTo2D(
      images.map((e) => e.embedding),
      (epoch, total) => setStatus(`Projecting — epoch ${epoch}/${total}…`),
      {
        nNeighbors: layoutSettings.nNeighbors,
        minDist: layoutSettings.minDist,
        nEpochs: layoutSettings.nEpochs,
        random: mulberry32(layoutSettings.seed),
      },
    )

    setStatus(`Arranging grid…`)

    // Grid assignment
    const aspect = window.innerWidth / window.innerHeight
    const { cells, cols, rows } = assignGrid(points, aspect)

    const items: ImageMapItem[] = cells.map((cell) => ({
      index: cell.index,
      name: images[cell.index].name,
      file: images[cell.index].file,
      col: cell.col,
      row: cell.row,
    }))

    setMapState({ items, cols, rows })
    setStatus(`${images.length} images`)
    setWorking(false)
  }

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

    let cachedCount = 0
    const images = await embedImages(files, (done, total, name, fromCache) => {
      if (fromCache) cachedCount++
      const cacheNote = cachedCount > 0 ? ` (${cachedCount} cached)` : ''
      setStatus(`Embedding ${done}/${total}${cacheNote}${name ? ` — ${name}` : ''}`)
    })

    if (images.length === 0) {
      setStatus('No images could be embedded')
      setWorking(false)
      return
    }

    setEmbedded(images)
    await runLayout(images, settings)
  }

  function handleSettingChange(patch: Partial<Omit<LayoutSettings, 'seed'>>) {
    setSettings((prev) => ({ ...prev, ...patch }))
  }

  function handleApplySettings() {
    if (embedded.length > 0) void runLayout(embedded, settings)
  }

  function handleRandomizeSeed() {
    const next = { ...settings, seed: randomSeed() }
    setSettings(next)
    if (embedded.length > 0) void runLayout(embedded, next)
  }

  const showMap = mapState !== null
  const hasEmbeddings = embedded.length > 0

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

        {ready && (
          <div className="settings">
            <label className="settings__row">
              <span>Neighbors</span>
              <input
                type="range"
                min={2}
                max={100}
                step={1}
                value={settings.nNeighbors}
                disabled={working}
                onChange={(e) => handleSettingChange({ nNeighbors: Number(e.target.value) })}
              />
              <span className="settings__value">{settings.nNeighbors}</span>
            </label>

            <label className="settings__row">
              <span>Min dist</span>
              <input
                type="range"
                min={0}
                max={0.99}
                step={0.01}
                value={settings.minDist}
                disabled={working}
                onChange={(e) => handleSettingChange({ minDist: Number(e.target.value) })}
              />
              <span className="settings__value">{settings.minDist.toFixed(2)}</span>
            </label>

            <label className="settings__row">
              <span>Epochs</span>
              <input
                type="range"
                min={50}
                max={1000}
                step={10}
                value={settings.nEpochs}
                disabled={working}
                onChange={(e) => handleSettingChange({ nEpochs: Number(e.target.value) })}
              />
              <span className="settings__value">{settings.nEpochs}</span>
            </label>

            <div className="settings__actions">
              <button type="button" disabled={!hasEmbeddings || working} onClick={handleApplySettings}>
                Apply
              </button>
              <button type="button" disabled={!hasEmbeddings || working} onClick={handleRandomizeSeed}>
                🎲 Randomize seed
              </button>
              <span className="settings__seed" title="Current UMAP seed">
                seed {settings.seed}
              </span>
            </div>
          </div>
        )}
      </header>

      {showMap && (
        <div className="map-container">
          <ImageMap items={mapState.items} cols={mapState.cols} rows={mapState.rows} />
        </div>
      )}
    </>
  )
}
