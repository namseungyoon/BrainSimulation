export interface CellTypeConfig {
  spatial?: {
    radius?: number
    count?: number
  }
  plotting?: {
    display_name?: string
    color?: string
  }
}

export interface NetworkDimensions {
  x?: number
  y?: number
  z?: number
}

export interface Ca1Config {
  name?: string
  network?: NetworkDimensions
  cell_types?: Record<string, CellTypeConfig>
}
