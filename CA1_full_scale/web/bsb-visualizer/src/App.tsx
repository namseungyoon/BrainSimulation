import { ChangeEvent, useCallback, useMemo, useState } from 'react'
import NetworkViewer, { type CellTypeDisplay } from './components/NetworkViewer'
import {
  loadNetworkFromHDF5,
  type BoundingBox,
  type RawCellPlacement,
  type Vec3,
} from './lib/h5Loader'
import type { Ca1Config } from './types/config'
import './App.css'

const COLOR_PALETTE = [
  '#E74C3C',
  '#2ECC71',
  '#3498DB',
  '#F1C40F',
  '#9B59B6',
  '#1ABC9C',
  '#E67E22',
  '#34495E',
  '#D35400',
  '#1F618D',
  '#B03A2E',
  '#117A65',
]

function App() {
  const [config, setConfig] = useState<Ca1Config | null>(null)
  const [rawPlacements, setRawPlacements] = useState<RawCellPlacement[]>([])
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null)
  const [networkDimensions, setNetworkDimensions] = useState<Vec3 | undefined>()
  const [visibility, setVisibility] = useState<Record<string, boolean>>({})
  const [status, setStatus] = useState('Brain Scaffold Builder에서 생성한 HDF5 파일을 불러오세요.')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [totalCellCount, setTotalCellCount] = useState(0)

  const handleConfigLoad = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }

    try {
      const text = await file.text()
      const parsed = JSON.parse(text) as Ca1Config
      setConfig(parsed)
      setStatus(`구성 파일 '${file.name}'을 불러왔습니다.`)
      setError(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setError(`구성 파일을 파싱할 수 없습니다: ${message}`)
    }
  }, [])

  const handleNetworkLoad = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      event.target.value = ''
      if (!file) {
        return
      }

      setIsLoading(true)
      setError(null)
      setStatus('HDF5 파일을 로딩 중입니다...')

      try {
        const result = await loadNetworkFromHDF5(file, config ?? undefined)
        setRawPlacements(result.cellPlacements)
        setBoundingBox(result.boundingBox)
        setNetworkDimensions(result.networkDimensions)
        setTotalCellCount(result.totalCellCount)

        const nextVisibility: Record<string, boolean> = {}
        for (const cell of result.cellPlacements) {
          nextVisibility[cell.name] = true
        }
        setVisibility(nextVisibility)

        setStatus(`'${file.name}' 로드 완료 (총 ${result.totalCellCount.toLocaleString()} 개의 셀)`) 
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        setStatus('HDF5 파일을 불러오지 못했습니다.')
        setRawPlacements([])
        setBoundingBox(null)
        setNetworkDimensions(undefined)
        setTotalCellCount(0)
        setVisibility({})
      } finally {
        setIsLoading(false)
      }
    },
    [config],
  )

  const cellDisplays = useMemo<CellTypeDisplay[]>(() => {
    if (!rawPlacements.length) {
      return []
    }

    const paletteAssignments = new Map<string, string>()
    let paletteIndex = 0

    return rawPlacements
      .map((placement) => {
        const cellConfig = config?.cell_types?.[placement.name]
        const color = (() => {
          const configured = cellConfig?.plotting?.color?.trim()
          if (configured) {
            return configured
          }
          if (!paletteAssignments.has(placement.name)) {
            const swatch = COLOR_PALETTE[paletteIndex % COLOR_PALETTE.length]
            paletteAssignments.set(placement.name, swatch)
            paletteIndex += 1
          }
          return paletteAssignments.get(placement.name) ?? COLOR_PALETTE[0]
        })()

        const radius = cellConfig?.spatial?.radius ?? 4
        const displayName = cellConfig?.plotting?.display_name ?? placement.name

        return {
          name: placement.name,
          displayName,
          color,
          radius,
          positions: placement.positions,
          count: placement.count,
        }
      })
      .sort((a, b) => b.count - a.count)
  }, [rawPlacements, config])

  const toggleCellVisibility = useCallback((cellName: string) => {
    setVisibility((prev) => ({
      ...prev,
      [cellName]: !(prev[cellName] ?? true),
    }))
  }, [])

  const setAllVisibility = useCallback((visible: boolean) => {
    setVisibility((prev) => {
      const next: Record<string, boolean> = {}
      for (const cell of cellDisplays) {
        next[cell.name] = visible
      }
      return next
    })
  }, [cellDisplays])

  const visibleCellCount = useMemo(() => {
    return cellDisplays.reduce((sum, cell) => {
      if (visibility[cell.name] ?? true) {
        return sum + cell.count
      }
      return sum
    }, 0)
  }, [cellDisplays, visibility])

  const networkSizeText = (() => {
    if (!networkDimensions) {
      return undefined
    }
    const { x, y, z } = networkDimensions
    if ([x, y, z].some((value) => value === undefined)) {
      return undefined
    }
    return `${x.toLocaleString()} x ${y.toLocaleString()} x ${z.toLocaleString()} um`
  })()

  return (
    <div className="app-root">
      <aside className="sidebar">
        <header>
          <h1>CA1 네트워크 뷰어</h1>
          <p className="subtitle">Brain Scaffold Builder HDF5를 three.js로 시각화합니다.</p>
        </header>

        <section className="panel">
          <h2>파일 불러오기</h2>
          <label className="file-input">
            <span>HDF5 파일 선택</span>
            <input type="file" accept=".h5,.hdf5,.hdf" onChange={handleNetworkLoad} />
          </label>
          <label className="file-input">
            <span>구성 JSON (선택)</span>
            <input type="file" accept=".json" onChange={handleConfigLoad} />
          </label>
          <p className="status">{status}</p>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel">
          <h2>네트워크 정보</h2>
          <ul className="info-list">
            <li>
              <strong>총 셀</strong>
              <span>{totalCellCount ? totalCellCount.toLocaleString() : '-'}</span>
            </li>
            <li>
              <strong>표시 중</strong>
              <span>{visibleCellCount ? visibleCellCount.toLocaleString() : '-'}</span>
            </li>
            {boundingBox && (
              <li>
                <strong>좌표 범위</strong>
                <span>
                  x {boundingBox.min.x.toFixed(1)}~{boundingBox.max.x.toFixed(1)}, y {boundingBox.min.y.toFixed(1)}~
                  {boundingBox.max.y.toFixed(1)}, z {boundingBox.min.z.toFixed(1)}~{boundingBox.max.z.toFixed(1)}
                </span>
              </li>
            )}
            {networkSizeText && (
              <li>
                <strong>모델 크기</strong>
                <span>{networkSizeText}</span>
              </li>
            )}
          </ul>

          <div className="visibility-controls">
            <button type="button" onClick={() => setAllVisibility(true)} disabled={!cellDisplays.length}>
              전체 표시
            </button>
            <button type="button" onClick={() => setAllVisibility(false)} disabled={!cellDisplays.length}>
              전체 숨김
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>셀 타입</h2>
          {cellDisplays.length === 0 ? (
            <p className="empty">로딩된 셀 타입이 없습니다.</p>
          ) : (
            <ul className="cell-list">
              {cellDisplays.map((cell) => {
                const checked = visibility[cell.name] ?? true
                return (
                  <li key={cell.name}>
                    <label>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleCellVisibility(cell.name)}
                      />
                      <span className="swatch" style={{ backgroundColor: cell.color }} />
                      <span className="cell-name">{cell.displayName}</span>
                      <span className="cell-count">{cell.count.toLocaleString()}</span>
                    </label>
                  </li>
                )
              })}
            </ul>
          )}
        </section>
      </aside>

      <main className="viewer-area">
        {rawPlacements.length ? (
          <NetworkViewer cellTypes={cellDisplays} visibility={visibility} boundingBox={boundingBox} />
        ) : (
          <div className="placeholder">{isLoading ? 'HDF5 파일을 불러오는 중입니다...' : '시각화할 데이터를 선택하세요.'}</div>
        )}
        {isLoading && <div className="loading-overlay">로딩 중...</div>}
      </main>
    </div>
  )
}

export default App
