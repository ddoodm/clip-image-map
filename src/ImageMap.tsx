import { useEffect, useLayoutEffect, useRef, useCallback } from 'react'

export type ImageMapItem = {
  index: number
  name: string
  file: File
  col: number
  row: number
}

type Props = {
  items: ImageMapItem[]
  cols: number
  rows: number
}

const CELL_PX = 128          // world-space cell size in pixels
const GAP_PX = 2             // gap between cells
const PLACEHOLDER_COLOR = '#2a2a2a'
const LABEL_FONT = '11px system-ui, sans-serif'
const MIN_SCALE = 0.05
const MAX_SCALE = 8

export function ImageMap({ items, cols, rows }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Camera state — mutable refs so we don't trigger re-renders on every pan/zoom
  const camRef = useRef({ x: 0, y: 0, scale: 1 })

  // Bitmap cache: index -> ImageBitmap | 'loading' | 'error'
  const bitmapCache = useRef<Map<number, ImageBitmap | 'loading' | 'error'>>(new Map())

  // Tooltip state
  const tooltipRef = useRef<{ name: string; x: number; y: number } | null>(null)

  const rafRef = useRef<number>(0)
  const dirty = useRef(true)

  const scheduleRedraw = useCallback(() => {
    dirty.current = true
  }, [])

  // ── Drawing ────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!dirty.current) return
    dirty.current = false

    const { x: ox, y: oy, scale } = camRef.current
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)

    // Viewport bounds in world-space for culling
    const worldLeft = -ox / scale
    const worldTop = -oy / scale
    const worldRight = worldLeft + W / scale
    const worldBottom = worldTop + H / scale

    ctx.save()
    ctx.translate(ox, oy)
    ctx.scale(scale, scale)

    for (const item of items) {
      const wx = item.col * (CELL_PX + GAP_PX)
      const wy = item.row * (CELL_PX + GAP_PX)

      // Cull
      if (wx + CELL_PX < worldLeft || wx > worldRight) continue
      if (wy + CELL_PX < worldTop || wy > worldBottom) continue

      const bmp = bitmapCache.current.get(item.index)

      if (bmp && bmp !== 'loading' && bmp !== 'error') {
        ctx.drawImage(bmp, wx, wy, CELL_PX, CELL_PX)
      } else {
        ctx.fillStyle = PLACEHOLDER_COLOR
        ctx.fillRect(wx, wy, CELL_PX, CELL_PX)

        // Kick off decode if not already started
        if (!bitmapCache.current.has(item.index)) {
          bitmapCache.current.set(item.index, 'loading')
          createImageBitmap(item.file, { resizeWidth: CELL_PX, resizeHeight: CELL_PX, resizeQuality: 'medium' })
            .then((bmp) => {
              bitmapCache.current.set(item.index, bmp)
              scheduleRedraw()
            })
            .catch(() => {
              bitmapCache.current.set(item.index, 'error')
            })
        }

        if (bmp === 'error') {
          ctx.fillStyle = '#555'
          ctx.fillRect(wx + CELL_PX / 2 - 10, wy + CELL_PX / 2 - 10, 20, 20)
        }
      }
    }

    ctx.restore()

    // Tooltip overlay (in screen space)
    const tip = tooltipRef.current
    if (tip && scale > 0.4) {
      ctx.save()
      ctx.font = LABEL_FONT
      const textW = ctx.measureText(tip.name).width + 12
      const textH = 20
      const tx = Math.min(tip.x + 12, W - textW - 4)
      const ty = Math.max(tip.y - 8, 4)
      ctx.fillStyle = 'rgba(0,0,0,0.7)'
      ctx.beginPath()
      ctx.roundRect(tx, ty, textW, textH, 4)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.fillText(tip.name, tx + 6, ty + 14)
      ctx.restore()
    }
  }, [items, scheduleRedraw])

  // ── RAF loop ───────────────────────────────────────────────────────────────

  useEffect(() => {
    let running = true
    function loop() {
      if (!running) return
      draw()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      running = false
      cancelAnimationFrame(rafRef.current)
    }
  }, [draw])

  // ── Resize observer ────────────────────────────────────────────────────────

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.round(rect.width * devicePixelRatio)
      canvas.height = Math.round(rect.height * devicePixelRatio)
      scheduleRedraw()
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [scheduleRedraw])

  // Fit the entire grid into view when items change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || items.length === 0) return
    const rect = canvas.getBoundingClientRect()
    const W = rect.width * devicePixelRatio
    const H = rect.height * devicePixelRatio
    const gridW = cols * (CELL_PX + GAP_PX) - GAP_PX
    const gridH = rows * (CELL_PX + GAP_PX) - GAP_PX
    const scale = Math.min((W / gridW) * 0.92, (H / gridH) * 0.92)
    const x = (W - gridW * scale) / 2
    const y = (H - gridH * scale) / 2
    camRef.current = { x, y, scale }
    scheduleRedraw()
  }, [items, cols, rows, scheduleRedraw])

  // ── Pointer events ─────────────────────────────────────────────────────────

  const dragRef = useRef<{ startX: number; startY: number; camX: number; camY: number } | null>(null)

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      camX: camRef.current.x,
      camY: camRef.current.y,
    }
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (dragRef.current) {
      const dx = (e.clientX - dragRef.current.startX) * devicePixelRatio
      const dy = (e.clientY - dragRef.current.startY) * devicePixelRatio
      camRef.current.x = dragRef.current.camX + dx
      camRef.current.y = dragRef.current.camY + dy
      scheduleRedraw()
      tooltipRef.current = null
      return
    }

    // Hover: find which cell is under the pointer
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = (e.clientX - rect.left) * devicePixelRatio
    const py = (e.clientY - rect.top) * devicePixelRatio
    const { x: ox, y: oy, scale } = camRef.current
    const wx = (px - ox) / scale
    const wy = (py - oy) / scale
    const col = Math.floor(wx / (CELL_PX + GAP_PX))
    const row = Math.floor(wy / (CELL_PX + GAP_PX))
    const cx = wx - col * (CELL_PX + GAP_PX)
    const cy = wy - row * (CELL_PX + GAP_PX)
    if (cx >= 0 && cx <= CELL_PX && cy >= 0 && cy <= CELL_PX) {
      const item = items.find((it) => it.col === col && it.row === row)
      if (item) {
        tooltipRef.current = { name: item.name, x: px, y: py }
        scheduleRedraw()
        return
      }
    }
    if (tooltipRef.current) {
      tooltipRef.current = null
      scheduleRedraw()
    }
  }, [items, scheduleRedraw])

  const onPointerUp = useCallback(() => {
    dragRef.current = null
  }, [])

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = (e.clientX - rect.left) * devicePixelRatio
    const py = (e.clientY - rect.top) * devicePixelRatio
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1
    const cam = camRef.current
    const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cam.scale * factor))
    const ratio = newScale / cam.scale
    cam.x = px - ratio * (px - cam.x)
    cam.y = py - ratio * (py - cam.y)
    cam.scale = newScale
    scheduleRedraw()
  }, [scheduleRedraw])

  // ── Cleanup bitmaps ────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      for (const v of bitmapCache.current.values()) {
        if (v !== 'loading' && v !== 'error') v.close()
      }
      bitmapCache.current.clear()
    }
  }, [items])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', width: '100%', height: '100%', cursor: dragRef.current ? 'grabbing' : 'grab' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
      onWheel={onWheel}
    />
  )
}
