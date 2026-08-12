import h5wasm, { Dataset, Group, Entity } from 'h5wasm'
import type { Ca1Config } from '../types/config'

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface BoundingBox {
  min: Vec3
  max: Vec3
  size: Vec3
  center: Vec3
}

export interface RawCellPlacement {
  name: string
  positions: Float32Array
  count: number
}

export interface NetworkLoadResult {
  cellPlacements: RawCellPlacement[]
  totalCellCount: number
  boundingBox: BoundingBox | null
  networkDimensions?: Vec3
}

const WORKDIR = '/bsb-files'

export async function loadNetworkFromHDF5(
  file: File,
  config?: Ca1Config,
): Promise<NetworkLoadResult> {
  const module = await h5wasm.ready
  const FS = module.FS

  ensureDirectory(FS, WORKDIR)

  const filePath = `${WORKDIR}/${Date.now()}_${sanitizeFileName(file.name)}`
  const buffer = new Uint8Array(await file.arrayBuffer())
  FS.writeFile(filePath, buffer)

  const h5File = new h5wasm.File(filePath, 'r')

  try {
    const placementEntity = h5File.get('placement')
    if (!isGroup(placementEntity)) {
      throw new Error('HDF5 파일에서 placement 그룹을 찾을 수 없습니다.')
    }

    const cellPlacements: RawCellPlacement[] = []

    for (const cellTypeName of placementEntity.keys()) {
      const cellEntity = placementEntity.get(cellTypeName)
      if (!isGroup(cellEntity)) {
        continue
      }

      const chunkArrays = collectPositionChunks(cellEntity)
      if (!chunkArrays.length) {
        continue
      }

      const merged = mergeFloat32Arrays(chunkArrays)
      const count = merged.length / 3
      cellPlacements.push({
        name: cellTypeName,
        positions: merged,
        count,
      })
    }

    const totalCellCount = cellPlacements.reduce((sum, cell) => sum + cell.count, 0)
    const boundingBox = computeBoundingBox(cellPlacements)

    const networkDimensions = config?.network
      ? {
          x: config.network.x ?? boundingBox?.size.x ?? 0,
          y: config.network.y ?? boundingBox?.size.y ?? 0,
          z: config.network.z ?? boundingBox?.size.z ?? 0,
        }
      : undefined

    return {
      cellPlacements,
      totalCellCount,
      boundingBox,
      networkDimensions,
    }
  } finally {
    h5File.close()
    try {
      FS.unlink(filePath)
    } catch (_error) {
      // 파일이 이미 제거되었거나 존재하지 않을 수 있으므로 무시
    }
  }
}

function collectPositionChunks(group: Group, depth = 0): Float32Array[] {
  const arrays: Float32Array[] = []

  for (const key of group.keys()) {
    const entity = group.get(key)
    if (!entity) {
      continue
    }

    if (isDataset(entity) && looksLikePositionDataset(entity, key)) {
      const array = datasetToFloat32(entity)
      if (array) {
        arrays.push(array)
      }
      continue
    }

    if (isGroup(entity) && depth < 3) {
      const nestedArrays = collectPositionChunks(entity, depth + 1)
      if (nestedArrays.length) {
        arrays.push(...nestedArrays)
      }
    }
  }

  return arrays
}

function looksLikePositionDataset(dataset: Dataset, key: string): boolean {
  const nameHint = key.toLowerCase()
  if (nameHint.includes('position') || nameHint === 'points') {
    const shape = dataset.shape
    return Array.isArray(shape) && shape.length === 2 && shape[1] === 3
  }

  const shape = dataset.shape
  return Array.isArray(shape) && shape.length === 2 && shape[1] === 3
}

function datasetToFloat32(dataset: Dataset): Float32Array | null {
  const value = dataset.value
  if (!value) {
    return null
  }

  return toFloat32Array(value)
}

function toFloat32Array(value: unknown): Float32Array | null {
  if (ArrayBuffer.isView(value)) {
    if (value instanceof Float32Array) {
      return value.slice()
    }

    if (
      value instanceof Float64Array ||
      value instanceof Int32Array ||
      value instanceof Uint32Array ||
      value instanceof Int16Array ||
      value instanceof Uint16Array ||
      value instanceof Int8Array ||
      value instanceof Uint8Array ||
      value instanceof Uint8ClampedArray
    ) {
      return Float32Array.from(value as unknown as Iterable<number>)
    }
  }

  if (Array.isArray(value)) {
    const flattened = value.flat(Infinity) as number[]
    return Float32Array.from(flattened)
  }

  return null
}

function mergeFloat32Arrays(chunks: Float32Array[]): Float32Array {
  if (chunks.length === 1) {
    return chunks[0]
  }

  const totalLength = chunks.reduce((sum, arr) => sum + arr.length, 0)
  const merged = new Float32Array(totalLength)

  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.length
  }

  return merged
}

function computeBoundingBox(cellPlacements: RawCellPlacement[]): BoundingBox | null {
  if (!cellPlacements.length) {
    return null
  }

  const min = { x: Infinity, y: Infinity, z: Infinity }
  const max = { x: -Infinity, y: -Infinity, z: -Infinity }

  for (const cell of cellPlacements) {
    const { positions } = cell
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]
      const y = positions[i + 1]
      const z = positions[i + 2]

      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        continue
      }

      if (x < min.x) min.x = x
      if (y < min.y) min.y = y
      if (z < min.z) min.z = z

      if (x > max.x) max.x = x
      if (y > max.y) max.y = y
      if (z > max.z) max.z = z
    }
  }

  if (!Number.isFinite(min.x)) {
    return null
  }

  const size = {
    x: max.x - min.x,
    y: max.y - min.y,
    z: max.z - min.z,
  }

  const center = {
    x: min.x + size.x / 2,
    y: min.y + size.y / 2,
    z: min.z + size.z / 2,
  }

  return { min, max, size, center }
}

function ensureDirectory(FS: any, path: string) {
  const parts = path.split('/').filter(Boolean)
  let current = ''

  for (const part of parts) {
    current += `/${part}`
    try {
      FS.mkdir(current)
    } catch (error: any) {
      if (error?.code === 'EEXIST') {
        continue
      }
      // 디렉터리가 이미 존재하는 경우가 아니면 오류를 다시 던진다
      if (!String(error).includes('File exists')) {
        throw error
      }
    }
  }
}

function sanitizeFileName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_.-]+/g, '_')
}

function isGroup(entity: Entity | null): entity is Group {
  return !!entity && entity instanceof Group
}

function isDataset(entity: Entity | null): entity is Dataset {
  return !!entity && entity instanceof Dataset
}
