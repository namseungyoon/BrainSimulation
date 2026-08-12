# ModelDB topology visualization

Generated from `configs/full_scale.yaml` with seed `12345` on the CPU. The representative
geometry uses scale `1` for all nine CA1 counts and for the context-only
CA3/ECIII counts. Positions and topology are not spatially cropped: every paired
panel uses the full 4000 × 1000 × 354 µm sheet and every incoming recurrent
biological projection for its highlighted post cell.

The population context displays a deterministic maximum of
`800` points per type. The paired panels do not
subsample edges. Split receptor ports and co-release rows are grouped into one
biological pre→post plan before sampling.

Posts are selected nearest the sheet center (x=2000, y=500 µm) and the median
depth plane. Neurogliaform uses its shallow SLM plane (z=187 µm), adjacent to
its Ivy/O-LM source layers, so the comparison reflects topology rather than a
deep-sheet feasibility boundary effect.

## Representative posts and measured locality

| post type | network index | position µm (x, y, z) | incoming edges/topology | uniform inner ring | 3-D inner ring |
|---|---:|---:|---:|---:|---:|
| Pyramidal | 165,217 | (1995.0, 495.0, 53.5) | 309 | 5.5% | 87.1% |
| Neurogliaform | 1,800 | (1987.5, 527.0, 187.0) | 58 | 5.2% | 84.5% |
| PV_Basket | 2,808 | (1989.0, 512.5, 29.0) | 524 | 4.8% | 86.8% |
| Bistratified | 1,130 | (1995.0, 493.0, 54.0) | 466 | 8.8% | 86.7% |

The old fixed-indegree edge identities are deterministic CPU samples without
replacement from the exact interval returned by
`binned_fixed_indegree_connections`; NEST-GPU normally samples that interval at
runtime. The new identities come directly from `ModelDbFastconn3D.iter_post_edges`.
For fast extraction, the exact representative position is presented to that
generator as a singleton target view (post index 0), while the old x-bin is
selected using the representative cell's network index.

## Color and size convention

Marker size is a categorical visual convention and does not encode abundance.

| type | full count | generated count | context shown | color | marker size |
|---|---:|---:|---:|---|---:|
| Pyramidal | 311,500 | 311,500 | 800 | #4E79A7 | 6.0 |
| PV_Basket | 5,530 | 5,530 | 800 | #E15759 | 9.0 |
| Bistratified | 2,210 | 2,210 | 800 | #EDC948 | 8.0 |
| O_LM | 1,640 | 1,640 | 800 | #59A14F | 8.0 |
| Axo | 1,470 | 1,470 | 800 | #B07AA1 | 8.5 |
| CCK_Basket | 3,600 | 3,600 | 800 | #FF9DA7 | 8.5 |
| Ivy | 8,810 | 8,810 | 800 | #9C755F | 7.5 |
| Neurogliaform | 3,580 | 3,580 | 800 | #76B7B2 | 8.0 |
| SCA | 400 | 400 | 400 | #79706E | 7.5 |
| CA3 | 204,700 | 204,700 | 800 | #F28E2B | 5.0 |
| ECIII | 250,000 | 250,000 | 800 | #2F9ED1 | 5.0 |

## Generator provenance

- `src/ca1/config.py::build_network_spec` — canonical full-scale counts/projections.
- `src/ca1/sim/modeldb_positions.py::modeldb_connectivity_positions` — positions,
  including external source populations.
- `src/ca1/sim/modeldb_positions.py::_positions_for_count` — ModelDB 3-D grid.
- `src/ca1/sim/modeldb_topology.py::binned_fixed_indegree_connections` — old
  uniform-x source-window topology.
- `src/ca1/sim/modeldb_topology.py::ModelDbFastconn3D.iter_post_edges` — new
  source-faithful 3-D Gaussian topology.

## Reading the figure

The topology change leaves the full sheet and indegree budgets intact but moves
presynaptic sources from a broad longitudinal window into a tight 3-D
neighborhood around each post cell (the innermost Gaussian ring is approximately
87% after the change).

Each HTML is self-contained Plotly. PNGs are high-resolution Matplotlib CPU
renders of the same data and camera views, used because this headless host blocks
the Kaleido/Chrome crash-handler sandbox.
