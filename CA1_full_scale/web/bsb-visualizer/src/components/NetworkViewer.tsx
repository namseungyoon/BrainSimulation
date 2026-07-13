import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { BoundingBox } from '../lib/h5Loader'

export interface CellTypeDisplay {
  name: string
  displayName: string
  color: string
  radius: number
  positions: Float32Array
  count: number
}

interface NetworkViewerProps {
  cellTypes: CellTypeDisplay[]
  visibility: Record<string, boolean>
  boundingBox: BoundingBox | null
}

const MIN_POINT_SIZE = 2

export function NetworkViewer({ cellTypes, visibility, boundingBox }: NetworkViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const animationRef = useRef<number | null>(null)
  const pointsRef = useRef<Record<string, THREE.Points>>({})
  const boundingBoxHelperRef = useRef<THREE.Box3Helper | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) {
      return
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x050509)
    sceneRef.current = scene

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setPixelRatio(window.devicePixelRatio || 1)
    rendererRef.current = renderer
    mount.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100000)
    cameraRef.current = camera

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controlsRef.current = controls

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6)
    directionalLight.position.set(0.5, 1, 0.8)
    scene.add(directionalLight)

    const axesHelper = new THREE.AxesHelper(100)
    axesHelper.material.depthTest = false
    axesHelper.renderOrder = 1
    scene.add(axesHelper)

    const resize = () => {
      const { clientWidth, clientHeight } = mount
      const width = Math.max(clientWidth, 1)
      const height = Math.max(clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }

    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(mount)
    resize()

    const renderLoop = () => {
      controls.update()
      renderer.render(scene, camera)
      animationRef.current = requestAnimationFrame(renderLoop)
    }
    animationRef.current = requestAnimationFrame(renderLoop)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      resizeObserver.disconnect()

      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)

      scene.traverse((object) => {
        if (object instanceof THREE.Points) {
          object.geometry.dispose()
          if (Array.isArray(object.material)) {
            object.material.forEach((mat) => mat.dispose())
          } else {
            object.material.dispose()
          }
        }
        if (object instanceof THREE.Box3Helper) {
          object.geometry.dispose()
          object.material.dispose()
        }
      })

      scene.clear()
      sceneRef.current = null
      rendererRef.current = null
      cameraRef.current = null
      controlsRef.current = null
      pointsRef.current = {}
      boundingBoxHelperRef.current = null
    }
  }, [])

  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) {
      return
    }

    if (boundingBoxHelperRef.current) {
      scene.remove(boundingBoxHelperRef.current)
      boundingBoxHelperRef.current.geometry.dispose()
      boundingBoxHelperRef.current.material.dispose()
      boundingBoxHelperRef.current = null
    }

    if (!boundingBox) {
      return
    }

    const box = new THREE.Box3(
      new THREE.Vector3(boundingBox.min.x, boundingBox.min.y, boundingBox.min.z),
      new THREE.Vector3(boundingBox.max.x, boundingBox.max.y, boundingBox.max.z),
    )

    const helper = new THREE.Box3Helper(box, 0x444444)
    helper.renderOrder = 0
    scene.add(helper)
    boundingBoxHelperRef.current = helper

    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!camera || !controls) {
      return
    }

    const maxSize = Math.max(boundingBox.size.x, boundingBox.size.y, boundingBox.size.z)
    const distance = Math.max(maxSize * 1.6, 200)

    camera.position.set(
      boundingBox.center.x + distance,
      boundingBox.center.y + distance,
      boundingBox.center.z + distance,
    )
    camera.near = Math.max(distance / 500, 0.1)
    camera.far = distance * 12
    camera.updateProjectionMatrix()

    controls.target.set(boundingBox.center.x, boundingBox.center.y, boundingBox.center.z)
    controls.update()
  }, [boundingBox])

  useEffect(() => {
    const scene = sceneRef.current
    const renderer = rendererRef.current
    const camera = cameraRef.current
    if (!scene || !renderer || !camera) {
      return
    }

    Object.values(pointsRef.current).forEach((points) => {
      scene.remove(points)
      points.geometry.dispose()
      if (Array.isArray(points.material)) {
        points.material.forEach((mat) => mat.dispose())
      } else {
        points.material.dispose()
      }
    })
    pointsRef.current = {}

    for (const cell of cellTypes) {
      if (!cell.positions || cell.positions.length === 0) {
        continue
      }

      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(cell.positions, 3))
      geometry.computeBoundingSphere()

      const pointSize = Math.max(cell.radius * 1.5, MIN_POINT_SIZE)

      let color: THREE.Color
      try {
        color = new THREE.Color(cell.color)
      } catch (_error) {
        color = new THREE.Color('#ffffff')
      }

      const material = new THREE.PointsMaterial({
        color,
        size: pointSize,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      })

      const points = new THREE.Points(geometry, material)
      points.name = cell.name
      points.visible = true
      scene.add(points)
      pointsRef.current[cell.name] = points
    }

    renderer.render(scene, camera)
  }, [cellTypes])

  useEffect(() => {
    const renderer = rendererRef.current
    const scene = sceneRef.current
    const camera = cameraRef.current
    if (!scene || !renderer || !camera) {
      return
    }

    for (const [name, visible] of Object.entries(visibility)) {
      const points = pointsRef.current[name]
      if (points) {
        points.visible = visible
      }
    }

    renderer.render(scene, camera)
  }, [visibility])

  const visibleCellCount = useMemo(() => {
    return cellTypes.reduce((sum, cell) => {
      if (visibility[cell.name] ?? true) {
        return sum + cell.count
      }
      return sum
    }, 0)
  }, [cellTypes, visibility])

  return (
    <div className="viewer-root">
      <div ref={mountRef} className="viewer-canvas" />
      <div className="viewer-overlay">
        <span>{`표시 중인 셀: ${visibleCellCount.toLocaleString()} 개`}</span>
      </div>
    </div>
  )
}

export default NetworkViewer
