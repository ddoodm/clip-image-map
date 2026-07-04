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

// Momentum tuning — kept subtle so the map still feels precise, with just a
// touch of glide after a pan release or a burst of scrolling.
const PAN_FRICTION = 0.9          // velocity retained per ~16.7ms frame
const PAN_STOP_SPEED = 0.02       // device px/ms below which pan velocity snaps to 0
const MAX_PAN_VELOCITY = 5        // device px/ms cap, to avoid huge flings on pointer jumps
const ZOOM_FRICTION = 0.85        // log-scale velocity retained per ~16.7ms frame
const ZOOM_STOP_SPEED = 0.00005   // log-scale units/ms below which zoom velocity snaps to 0
const ZOOM_VELOCITY_GAIN = 0.5    // how much of each wheel tick feeds into momentum
const MAX_ZOOM_VELOCITY = 0.005   // log-scale units/ms cap

// Decoding thousands of images at once (e.g. right after "fit to view" reveals
// the whole grid) can exceed the browser's concurrent-decode/memory budget and
// cause createImageBitmap to reject for a chunk of them. Cap concurrency and
// retry transient failures instead of giving up on the first rejection.
const MAX_CONCURRENT_DECODES = 12
const MAX_DECODE_RETRIES = 3
const DECODE_RETRY_BASE_MS = 300

export function ImageMap({ items, cols, rows }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Camera state — mutable refs so we don't trigger re-renders on every pan/zoom
  const camRef = useRef({ x: 0, y: 0, scale: 1 })

  // Bitmap cache: index -> ImageBitmap | 'loading' | 'error'
  const bitmapCache = useRef<Map<number, ImageBitmap | 'loading' | 'error'>>(new Map())

  // Decode queue + in-flight counter used to throttle concurrent createImageBitmap calls
  const decodeQueueRef = useRef<ImageMapItem[]>([])
  const inFlightRef = useRef(0)
  const retryCountRef = useRef<Map<number, number>>(new Map())

  // Tooltip state
  const tooltipRef = useRef<{ name: string; x: number; y: number } | null>(null)

  // Momentum state for pan (device px/ms) and zoom (log-scale units/ms, plus
  // the screen point to zoom around while it eases out).
  const panVelocityRef = useRef({ vx: 0, vy: 0 })
  const zoomVelocityRef = useRef({ vLog: 0, px: 0, py: 0 })

  const rafRef = useRef<number>(0)
  const dirty = useRef(true)

  const scheduleRedraw = useCallback(() => {
    dirty.current = true
  }, [])

  // ── Decoding ───────────────────────────────────────────────────────────────

  /**
   * Pulls items off the decode queue up to MAX_CONCURRENT_DECODES in flight.
   * On failure, retries with exponential backoff a few times before giving up
   * and marking the cell as 'error' — this is what lets transient
   * resource-exhaustion failures (rather than genuinely corrupt files) recover
   * instead of permanently falling back to the placeholder.
   */
  const pumpDecodeQueue = useCallback(() => {
    while (inFlightRef.current < MAX_CONCURRENT_DECODES && decodeQueueRef.current.length > 0) {
      const item = decodeQueueRef.current.shift()!
      inFlightRef.current++

      createImageBitmap(item.file, { resizeWidth: CELL_PX, resizeHeight: CELL_PX, resizeQuality: 'medium' })
        .then((bmp) => {
          bitmapCache.current.set(item.index, bmp)
          scheduleRedraw()
        })
        .catch((error) => {
          const retries = retryCountRef.current.get(item.index) ?? 0
          if (retries < MAX_DECODE_RETRIES) {
            retryCountRef.current.set(item.index, retries + 1)
            const delay = DECODE_RETRY_BASE_MS * 2 ** retries
            console.warn(`Decode failed for "${item.name}", retrying (${retries + 1}/${MAX_DECODE_RETRIES})…`, error)
            setTimeout(() => {
              decodeQueueRef.current.push(item)
              pumpDecodeQueue()
            }, delay)
          } else {
            console.error(`Giving up decoding "${item.name}" after ${MAX_DECODE_RETRIES} retries:`, error)
            bitmapCache.current.set(item.index, 'error')
            scheduleRedraw()
          }
        })
        .finally(() => {
          inFlightRef.current--
          pumpDecodeQueue()
        })
    }
  }, [scheduleRedraw])

  // ── Camera helpers ─────────────────────────────────────────────────────────

  const applyZoomFactor = useCallback((factor: number, px: number, py: number) => {
    const cam = camRef.current
    const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cam.scale * factor))
    const ratio = newScale / cam.scale
    cam.x = px - ratio * (px - cam.x)
    cam.y = py - ratio * (py - cam.y)
    cam.scale = newScale
  }, [])

  /**
   * Advances pan/zoom momentum by dt milliseconds. Pan only glides while the
   * pointer isn't actively dragging; zoom momentum always eases out, layering
   * on top of whatever the user is doing with the wheel.
   */
  const stepMomentum = useCallback((dt: number) => {
    let changed = false
    const frameDecay = dt / 16.6667

    if (!dragRef.current) {
      const pv = panVelocityRef.current
      if (Math.abs(pv.vx) > PAN_STOP_SPEED || Math.abs(pv.vy) > PAN_STOP_SPEED) {
        camRef.current.x += pv.vx * dt
        camRef.current.y += pv.vy * dt
        const decay = Math.pow(PAN_FRICTION, frameDecay)
        pv.vx *= decay
        pv.vy *= decay
        changed = true
      } else if (pv.vx !== 0 || pv.vy !== 0) {
        pv.vx = 0
        pv.vy = 0
      }
    }

    const zv = zoomVelocityRef.current
    if (Math.abs(zv.vLog) > ZOOM_STOP_SPEED) {
      applyZoomFactor(Math.exp(zv.vLog * dt), zv.px, zv.py)
      zv.vLog *= Math.pow(ZOOM_FRICTION, frameDecay)
      changed = true
    } else if (zv.vLog !== 0) {
      zv.vLog = 0
    }

    if (changed) scheduleRedraw()
  }, [applyZoomFactor, scheduleRedraw])

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

        // Queue decode if not already started (throttled — see pumpDecodeQueue)
        if (!bitmapCache.current.has(item.index)) {
          bitmapCache.current.set(item.index, 'loading')
          decodeQueueRef.current.push(item)
        }

        if (bmp === 'error') {
          ctx.fillStyle = '#555'
          ctx.fillRect(wx + CELL_PX / 2 - 10, wy + CELL_PX / 2 - 10, 20, 20)
        }
      }
    }

    pumpDecodeQueue()

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
  }, [items, scheduleRedraw, pumpDecodeQueue])

  // ── RAF loop ───────────────────────────────────────────────────────────────

  useEffect(() => {
    let running = true
    let lastT: number | null = null
    function loop(t: number) {
      if (!running) return
      // Clamp dt so returning from a backgrounded tab doesn't produce one huge jump.
      const dt = lastT === null ? 16.6667 : Math.min(64, t - lastT)
      lastT = t
      stepMomentum(dt)
      draw()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      running = false
      cancelAnimationFrame(rafRef.current)
    }
  }, [draw, stepMomentum])

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
  const lastMoveRef = useRef<{ x: number; y: number; t: number } | null>(null)

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      camX: camRef.current.x,
      camY: camRef.current.y,
    }
    panVelocityRef.current = { vx: 0, vy: 0 }
    lastMoveRef.current = { x: e.clientX * devicePixelRatio, y: e.clientY * devicePixelRatio, t: performance.now() }
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (dragRef.current) {
      const dx = (e.clientX - dragRef.current.startX) * devicePixelRatio
      const dy = (e.clientY - dragRef.current.startY) * devicePixelRatio
      camRef.current.x = dragRef.current.camX + dx
      camRef.current.y = dragRef.current.camY + dy

      // Track a smoothed release velocity so lifting the pointer glides on
      // instead of stopping dead.
      const now = performance.now()
      const last = lastMoveRef.current
      if (last) {
        const dt = Math.max(1, now - last.t)
        const px = e.clientX * devicePixelRatio
        const py = e.clientY * devicePixelRatio
        const rawVx = Math.max(-MAX_PAN_VELOCITY, Math.min(MAX_PAN_VELOCITY, (px - last.x) / dt))
        const rawVy = Math.max(-MAX_PAN_VELOCITY, Math.min(MAX_PAN_VELOCITY, (py - last.y) / dt))
        panVelocityRef.current.vx = panVelocityRef.current.vx * 0.6 + rawVx * 0.4
        panVelocityRef.current.vy = panVelocityRef.current.vy * 0.6 + rawVy * 0.4
        lastMoveRef.current = { x: px, y: py, t: now }
      }

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
    // Leave panVelocityRef as-is so the momentum step can glide it to a stop.
    dragRef.current = null
    lastMoveRef.current = null
  }, [])

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = (e.clientX - rect.left) * devicePixelRatio
    const py = (e.clientY - rect.top) * devicePixelRatio
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1
    applyZoomFactor(factor, px, py)

    // Build up a touch of zoom momentum so scrolling eases out rather than stopping dead.
    const zv = zoomVelocityRef.current
    const combined = zv.vLog + Math.log(factor) * ZOOM_VELOCITY_GAIN
    zv.vLog = Math.max(-MAX_ZOOM_VELOCITY, Math.min(MAX_ZOOM_VELOCITY, combined))
    zv.px = px
    zv.py = py

    scheduleRedraw()
  }, [applyZoomFactor, scheduleRedraw])

  // ── Cleanup bitmaps ────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      for (const v of bitmapCache.current.values()) {
        if (v !== 'loading' && v !== 'error') v.close()
      }
      bitmapCache.current.clear()
      decodeQueueRef.current = []
      retryCountRef.current.clear()
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
