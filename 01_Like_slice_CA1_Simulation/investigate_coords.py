import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import nrrd

# 1. Load slice_cells.npz and get xyz ranges
npz_path = "05_placement/slice_cells.npz"
data = np.load(npz_path)
print("NPZ keys:", list(data.keys()))

if 'xyz' in data:
    xyz = data['xyz']
    print(f"\nslice_cells.npz 'xyz' shape: {xyz.shape}, dtype: {xyz.dtype}")
    print(f"xyz min: {xyz.min(axis=0)}")
    print(f"xyz max: {xyz.max(axis=0)}")
    print(f"xyz range (max - min): {xyz.max(axis=0) - xyz.min(axis=0)}")
else:
    print("'xyz' key not found in NPZ. Available keys:", list(data.keys()))

# 2. Read brain_regions.nrrd header
brain_regions_path = "data/atlas/brain_regions.nrrd"
try:
    br_data, br_header = nrrd.read(brain_regions_path, header_only=True)
except Exception as e:
    print(f"Error reading brain_regions header: {e}")
    # Try without header_only
    br_data, br_header = nrrd.read(brain_regions_path)

print(f"\nbrain_regions.nrrd:")
print(f"  shape: {br_data.shape}")
print(f"  dtype: {br_data.dtype}")
print(f"  space: {br_header.get('space', 'N/A')}")
print(f"  space origin: {br_header.get('space origin', 'N/A')}")
print(f"  space directions:\n{br_header.get('space directions', 'N/A')}")
print(f"  sizes: {br_header.get('sizes', 'N/A')}")

# Calculate physical coordinate range
space_directions = br_header.get('space directions', None)
space_origin = br_header.get('space origin', None)
sizes = br_data.shape

if space_directions is not None and space_origin is not None:
    # Each axis has a direction vector
    origin = np.array(space_origin)
    directions = np.array(space_directions)
    
    # Calculate extent for each axis
    # For axis i: min = origin, max = origin + directions[i] * (sizes[i] - 1)
    print(f"\nPhysical coordinate calculation:")
    print(f"  origin (µm): {origin}")
    print(f"  directions (µm/voxel):\n{directions}")
    
    max_coords = origin + directions @ (np.array(sizes) - 1)
    print(f"  Physical max (origin + directions*(sizes-1)): {max_coords}")
    print(f"  Physical range:")
    for i, ax in enumerate(['x', 'y', 'z']):
        print(f"    {ax}: [{origin[i]:.2f}, {max_coords[i]:.2f}] µm")

# 3. Read CA1_SP.nrrd header
ca1_sp_path = "data/atlas/nrrd_volumes/CA1/CA1_SP.nrrd"
try:
    ca1_data, ca1_header = nrrd.read(ca1_sp_path, header_only=True)
except Exception as e:
    print(f"Error reading CA1_SP header: {e}")
    ca1_data, ca1_header = nrrd.read(ca1_sp_path)

print(f"\nCA1_SP.nrrd:")
print(f"  shape: {ca1_data.shape}")
print(f"  dtype: {ca1_data.dtype}")
print(f"  space: {ca1_header.get('space', 'N/A')}")
print(f"  space origin: {ca1_header.get('space origin', 'N/A')}")
print(f"  space directions:\n{ca1_header.get('space directions', 'N/A')}")

