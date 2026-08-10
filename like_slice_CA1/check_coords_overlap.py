import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import nrrd

# Load slice_cells.npz
npz_path = "05_placement/slice_cells.npz"
data = np.load(npz_path)
xyz = data['xyz']

print("="*60)
print("STEP 3: Coordinate System Alignment Check")
print("="*60)

print("\nslice_cells.npz xyz ranges:")
print(f"  x: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] µm")
print(f"  y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] µm")
print(f"  z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] µm")

# Atlas ranges (from previous output)
atlas_origin = np.array([-46.54, -152.16, -152.])
atlas_directions = np.array([[16., 0., 0.],
                              [0., 16., 0.],
                              [0., 0., 16.]])
atlas_sizes = np.array([308, 495, 464])
atlas_max = atlas_origin + atlas_directions @ (atlas_sizes - 1)

print("\nAtlas ranges:")
print(f"  x: [{atlas_origin[0]:.2f}, {atlas_max[0]:.2f}] µm")
print(f"  y: [{atlas_origin[1]:.2f}, {atlas_max[1]:.2f}] µm")
print(f"  z: [{atlas_origin[2]:.2f}, {atlas_max[2]:.2f}] µm")

# Check overlap
print("\nCoordinate overlap check:")
x_overlap = (xyz[:, 0].min() >= atlas_origin[0]) and (xyz[:, 0].max() <= atlas_max[0])
y_overlap = (xyz[:, 1].min() >= atlas_origin[1]) and (xyz[:, 1].max() <= atlas_max[1])
z_overlap = (xyz[:, 2].min() >= atlas_origin[2]) and (xyz[:, 2].max() <= atlas_max[2])

print(f"  x in atlas range: {x_overlap}")
print(f"  y in atlas range: {y_overlap}")
print(f"  z in atlas range: {z_overlap}")
print(f"  All axes in range: {x_overlap and y_overlap and z_overlap}")

# Check for out-of-bounds coordinates
out_of_bounds = np.any(xyz < atlas_origin, axis=1) | np.any(xyz > atlas_max, axis=1)
n_out_of_bounds = np.sum(out_of_bounds)
print(f"\n  Cells out of atlas bounds: {n_out_of_bounds} / {len(xyz)}")

if n_out_of_bounds > 0:
    print(f"    Out-of-bounds cells: {np.where(out_of_bounds)[0][:10]}... (showing first 10)")

print("\n" + "="*60)
print("STEP 4: Voxel Index Conversion Formula")
print("="*60)

print("\nFormula derivation:")
print("  Given:")
print("    - Physical point: p = [px, py, pz] (µm)")
print("    - Space origin: O = [-46.54, -152.16, -152.00] (µm)")
print("    - Space directions matrix (each row is voxel spacing):")
print("      D = [[16,  0,  0],")
print("           [ 0, 16,  0],")
print("           [ 0,  0, 16]]")
print("\n  Voxel index vector:")
print("    v = D^(-1) * (p - O)")
print("\n  Since D is diagonal with value 16:")
print("    v[i] = (p[i] - O[i]) / 16")
print("\n  Specifically:")
print("    v_x = (px - (-46.54)) / 16 = (px + 46.54) / 16")
print("    v_y = (py - (-152.16)) / 16 = (py + 152.16) / 16")
print("    v_z = (pz - (-152.00)) / 16 = (pz + 152.00) / 16")

# Convert xyz to voxel indices
voxel_indices = (xyz - atlas_origin) / 16.0

print("\n" + "="*60)
print("STEP 5: Verification with CA1_SP.nrrd Mask")
print("="*60)

# Load CA1_SP.nrrd
ca1_sp_path = "data/atlas/nrrd_volumes/CA1/CA1_SP.nrrd"
ca1_data, ca1_header = nrrd.read(ca1_sp_path)

print(f"\nCA1_SP.nrrd loaded: shape={ca1_data.shape}, dtype={ca1_data.dtype}")
print(f"CA1_SP value range: [{ca1_data.min()}, {ca1_data.max()}]")

# Round voxel indices to nearest integer for indexing
voxel_indices_int = np.round(voxel_indices).astype(int)

# Check which indices are valid (within bounds)
valid_indices = (voxel_indices_int[:, 0] >= 0) & (voxel_indices_int[:, 0] < ca1_data.shape[0]) & \
                (voxel_indices_int[:, 1] >= 0) & (voxel_indices_int[:, 1] < ca1_data.shape[1]) & \
                (voxel_indices_int[:, 2] >= 0) & (voxel_indices_int[:, 2] < ca1_data.shape[2])

print(f"\nVoxel indices within CA1_SP bounds: {np.sum(valid_indices)} / {len(xyz)}")

# For valid indices, check if CA1_SP value is 1
if np.sum(valid_indices) > 0:
    ca1_values = ca1_data[voxel_indices_int[valid_indices, 0],
                          voxel_indices_int[valid_indices, 1],
                          voxel_indices_int[valid_indices, 2]]
    
    cells_in_ca1 = np.sum(ca1_values == 1)
    cells_not_in_ca1 = np.sum(ca1_values != 1)
    
    print(f"\nCA1_SP mask validation:")
    print(f"  Cells with CA1_SP[voxel] == 1: {cells_in_ca1} ({100*cells_in_ca1/np.sum(valid_indices):.1f}%)")
    print(f"  Cells with CA1_SP[voxel] != 1: {cells_not_in_ca1} ({100*cells_not_in_ca1/np.sum(valid_indices):.1f}%)")
    print(f"  CA1_SP value distribution (in valid cells):")
    unique_vals, counts = np.unique(ca1_values, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        print(f"    {int(val)}: {cnt} cells")

print("\n" + "="*60)

