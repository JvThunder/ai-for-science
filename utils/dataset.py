"""
Multi-file streaming dataset for TB-scale molecular geometry data.

This module provides a streaming dataset implementation that:
- Handles multiple CSV files (TB-scale data)
- Streams data from disk with NO RAM caching
- Uses file-based epochs (each file = 1 epoch)
- Reads CSV in chunks for memory efficiency
- Integrates with stereochemistry-aware geometry computation

Key Classes:
- MultiFileStreamingDataset: Main streaming dataset for training
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from collections import defaultdict, deque, OrderedDict
from typing import List, Dict, Optional, Tuple
import json
import glob
import random
from tqdm import tqdm
import sys
import os
import psutil

# Import our geometry utilities
sys.path.append(os.path.dirname(__file__))
from geometry import compute_local_geometry


def parse_xyz_string(xyz_string: str) -> Tuple[List[List[float]], List[str]]:
    """
    Parse XYZ format string to extract atom coordinates and elements.

    Args:
        xyz_string: XYZ format string (possibly JSON-encoded)

    Returns:
        coordinates: List of [x, y, z] coordinates
        elements: List of element symbols
    """
    # Handle JSON encoding
    if isinstance(xyz_string, str):
        if xyz_string.startswith('"') and xyz_string.endswith('"'):
            try:
                xyz_string = json.loads(xyz_string)
            except json.JSONDecodeError:
                xyz_string = xyz_string[1:-1]
                xyz_string = xyz_string.replace('\\"', '"').replace('\\\\', '\\')
        xyz_string = xyz_string.replace('\\n', '\n')

    lines = xyz_string.strip().split('\n')
    if len(lines) < 3:
        return [], []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        return [], []

    coordinates = []
    elements = []

    for i in range(2, min(2 + num_atoms, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            element = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                coordinates.append([x, y, z])
                elements.append(element)
            except ValueError:
                continue

    return coordinates, elements


def parse_bonds_from_dataset(bonds_string) -> List[Dict]:
    """
    Parse bonds_geom_valence JSON string to extract bond information.

    Args:
        bonds_string: JSON string or list containing bond data

    Returns:
        List of bond dictionaries with 'i' and 'j' keys
    """
    if isinstance(bonds_string, str):
        if bonds_string.startswith('"') and bonds_string.endswith('"'):
            try:
                bonds_string = json.loads(bonds_string)
            except json.JSONDecodeError:
                bonds_string = bonds_string[1:-1]

        # Handle escaped JSON
        if isinstance(bonds_string, str):
            try:
                bonds_data = json.loads(bonds_string)
            except json.JSONDecodeError:
                return []
        else:
            bonds_data = bonds_string
    else:
        bonds_data = bonds_string

    return bonds_data if isinstance(bonds_data, list) else []


class ChunkCache:
    """
    LRU cache for CSV chunks with RAM limit.

    Keeps recently accessed chunks in memory to avoid repeated disk reads.
    Automatically evicts old chunks when RAM limit is reached.
    """

    def __init__(self, max_ram_mb: int = 4096):
        """
        Initialize chunk cache.

        Args:
            max_ram_mb: Maximum RAM to use in MB (default: 4GB)
        """
        self.max_ram_bytes = max_ram_mb * 1024 * 1024
        self.cache = OrderedDict()  # LRU cache: (file_path, chunk_idx) -> DataFrame
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def get(self, file_path: str, chunk_idx: int) -> Optional[pd.DataFrame]:
        """
        Get chunk from cache.

        Returns None if not in cache (cache miss).
        """
        key = (file_path, chunk_idx)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, file_path: str, chunk_idx: int, chunk: pd.DataFrame):
        """
        Add chunk to cache, evicting old chunks if needed.
        """
        key = (file_path, chunk_idx)

        # Estimate chunk size in bytes
        chunk_size = chunk.memory_usage(deep=True).sum()

        # Evict old chunks if needed
        while self.current_size + chunk_size > self.max_ram_bytes and len(self.cache) > 0:
            # Remove least recently used (first item)
            old_key, old_chunk = self.cache.popitem(last=False)
            old_size = old_chunk.memory_usage(deep=True).sum()
            self.current_size -= old_size

        # Add new chunk
        if chunk_size <= self.max_ram_bytes:  # Only cache if it fits
            self.cache[key] = chunk
            self.current_size += chunk_size

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.current_size = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'num_chunks': len(self.cache),
            'ram_usage_mb': self.current_size / (1024 * 1024)
        }


def dfs_subgraph(graph: Dict[int, List[int]], start_node: int, depth: int = 1) -> set:
    """
    Extract subgraph using DFS up to specified depth.

    Args:
        graph: Adjacency list representation
        start_node: Starting node
        depth: Maximum depth to explore

    Returns:
        Set of nodes in the subgraph
    """
    visited = set()
    queue = deque([(start_node, 0)])  # (node, current_depth)
    subgraph_nodes = set()

    while queue:
        node, curr_depth = queue.popleft()
        if node in visited or curr_depth > depth:
            continue

        visited.add(node)
        subgraph_nodes.add(node)

        if curr_depth < depth:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, curr_depth + 1))

    return subgraph_nodes


class MultiFileStreamingDataset(Dataset):
    """
    Streaming dataset for TB-scale molecular geometry data.

    Key Features:
    - NO RAM caching - streams everything from disk
    - File-based epochs: each file = 1 epoch
    - Chunked CSV reading for memory efficiency
    - Stereochemistry-aware geometry computation
    - Supports multiple file patterns (e.g., "Compound_*.csv")

    Usage:
        >>> dataset = MultiFileStreamingDataset(
        ...     file_pattern="../Compound_*.csv",
        ...     element_to_idx=element_mapping,
        ...     current_file_idx=0
        ... )
        >>> dataset.next_file()  # Cycle to next file
        >>> data = dataset[0]    # Stream one sample
    """

    def __init__(
        self,
        file_paths: List[str],
        element_to_idx: Dict[str, int],
        current_file_idx: int = 0,
        max_bonds: int = 7,
        bond_threshold: float = 4.0,
        subgraph_depth: int = 1,
        chunk_size: int = 10000,
        include_non_bonded: bool = False,
        training: bool = True,
        max_samples: Optional[int] = None,
        cache_ram_mb: int = 4096,
        val_split_ratio: float = 0.0,
        split: Optional[str] = None,
        split_seed: int = 42
    ):
        """
        Initialize multi-file streaming dataset.

        Args:
            file_paths: List of CSV file paths
            element_to_idx: Mapping from element symbols to indices
            current_file_idx: Index of current file to use (for file-based epochs)
            max_bonds: Maximum number of bonds per atom to predict
            bond_threshold: Distance threshold for non-bonded pairs (Angstroms)
            subgraph_depth: Depth for local subgraph extraction
            chunk_size: Number of rows to read at once from CSV
            include_non_bonded: Include non-bonded pairs for distance diversity
            training: If True, apply random rotation augmentation
            max_samples: Maximum number of samples (bonds) to use (for testing)
            cache_ram_mb: Maximum RAM for caching chunks (MB, default: 4GB)
            val_split_ratio: Fraction of data to use for validation (0.0-1.0)
            split: Which split to use - 'train', 'val', or None (use all data)
            split_seed: Random seed for train/val splitting (for reproducibility)
        """
        super().__init__()

        # Initialize chunk cache
        self.chunk_cache = ChunkCache(max_ram_mb=cache_ram_mb)

        # File management
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        if not self.file_paths:
            raise ValueError(f"No files provided")

        self.current_file_idx = current_file_idx % len(self.file_paths)
        self.current_file = self.file_paths[self.current_file_idx]

        # Dataset parameters
        self.element_to_idx = element_to_idx
        self.max_bonds = max_bonds
        self.bond_threshold = bond_threshold
        self.subgraph_depth = subgraph_depth
        self.chunk_size = chunk_size
        self.include_non_bonded = include_non_bonded
        self.training = training
        self.max_samples = max_samples
        self.val_split_ratio = val_split_ratio
        self.split = split
        self.split_seed = split_seed

        # Index mapping (lightweight - only counts, not data)
        self.valid_molecule_indices = []  # (chunk_idx, row_in_chunk, bond_count)
        self.total_bonds = 0

        # Load current file metadata
        self._load_file_metadata()

        print(f"Dataset initialized with {len(self.file_paths)} files")
        print(f"Current file: {self.current_file}")
        print(f"Found {len(self.valid_molecule_indices)} valid molecules with {self.total_bonds} bonds")

    def _load_file_metadata(self):
        """
        Load metadata for current file (molecule counts, bond counts).

        Does NOT load actual molecule data - only counts for indexing.
        Memory usage: O(num_molecules) not O(num_atoms * num_molecules)
        """
        print(f"\nLoading metadata from: {self.current_file}")

        all_molecule_indices = []
        chunk_idx = 0

        # Read CSV in chunks and collect all molecule indices
        for chunk in pd.read_csv(self.current_file, chunksize=self.chunk_size):
            for row_in_chunk, (_, row) in enumerate(chunk.iterrows()):
                try:
                    # Quick validation
                    if 'geometry_xyz' not in row or pd.isna(row['geometry_xyz']):
                        continue
                    if 'bonds_geom_valence' not in row or pd.isna(row['bonds_geom_valence']):
                        continue

                    # Parse to validate
                    coordinates, elements = parse_xyz_string(row['geometry_xyz'])
                    if not coordinates or len(coordinates) < 2:
                        continue

                    bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])
                    if not bonds_data:
                        continue

                    # Count bonds
                    bond_count = 0
                    for bond in bonds_data:
                        if 'i' in bond and 'j' in bond:
                            i, j = bond['i'], bond['j']
                            if i < len(coordinates) and j < len(coordinates) and i != j:
                                bond_count += 1

                    # Add non-bonded pairs count if requested
                    if self.include_non_bonded and bond_count > 0:
                        # Estimate non-bonded pairs (will sample 1/3 of bonds)
                        bond_count += bond_count // 3

                    if bond_count > 0:
                        all_molecule_indices.append((chunk_idx, row_in_chunk, bond_count))

                except Exception:
                    continue

            chunk_idx += 1

        # Apply train/val split if requested
        if self.val_split_ratio > 0.0 and self.split is not None:
            self.valid_molecule_indices = self._apply_train_val_split(all_molecule_indices)
        else:
            self.valid_molecule_indices = all_molecule_indices

        # Calculate total bonds
        self.total_bonds = sum(bond_count for _, _, bond_count in self.valid_molecule_indices)

        # Apply max_samples limit if specified
        if self.max_samples and self.total_bonds > self.max_samples:
            total = 0
            for idx, (chunk_idx, row_in_chunk, bond_count) in enumerate(self.valid_molecule_indices):
                total += bond_count
                if total >= self.max_samples:
                    self.valid_molecule_indices = self.valid_molecule_indices[:idx+1]
                    self.total_bonds = self.max_samples
                    break

        split_info = f" ({self.split} split)" if self.split else ""
        print(f"Metadata loaded{split_info}: {len(self.valid_molecule_indices)} molecules, {self.total_bonds} bonds")

    def _apply_train_val_split(self, all_molecule_indices: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Apply train/val split to molecule indices.

        Args:
            all_molecule_indices: List of (chunk_idx, row_in_chunk, bond_count) tuples

        Returns:
            Filtered list of molecule indices for the specified split
        """
        if not all_molecule_indices:
            return []

        # Use a deterministic random seed for reproducible splits
        rng = random.Random(self.split_seed + hash(self.current_file))

        # Shuffle indices with the deterministic seed
        shuffled_indices = list(all_molecule_indices)
        rng.shuffle(shuffled_indices)

        # Calculate split point
        n_total = len(shuffled_indices)
        n_val = int(n_total * self.val_split_ratio)
        n_train = n_total - n_val

        # Return appropriate split
        if self.split == 'train':
            return shuffled_indices[:n_train]
        elif self.split == 'val':
            return shuffled_indices[n_train:]
        else:
            return all_molecule_indices

    def next_file(self) -> str:
        """
        Cycle to the next file (for file-based epochs).

        Returns:
            Path to new current file
        """
        # Print cache stats before switching
        cache_stats = self.chunk_cache.get_stats()
        print(f"\nCache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
              f"hit rate: {cache_stats['hit_rate']:.2%}, RAM: {cache_stats['ram_usage_mb']:.1f} MB")

        # Clear cache when switching files to free memory
        self.chunk_cache.clear()

        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_paths)
        self.current_file = self.file_paths[self.current_file_idx]

        print(f"\nSwitching to file {self.current_file_idx + 1}/{len(self.file_paths)}: {self.current_file}")

        # Reload metadata for new file
        self._load_file_metadata()

        return self.current_file

    def get_current_file_info(self) -> Dict:
        """Get information about current file."""
        return {
            'file_idx': self.current_file_idx,
            'file_path': self.current_file,
            'total_files': len(self.file_paths),
            'num_molecules': len(self.valid_molecule_indices),
            'num_bonds': self.total_bonds
        }

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.chunk_cache.get_stats()
    def __len__(self) -> int:
        """Return total number of training samples (bonds) in current file."""
        if self.max_samples:
            return min(self.total_bonds, self.max_samples)
        return self.total_bonds

    def _get_molecule_and_bond_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Convert flat bond index to (chunk_idx, row_in_chunk, bond_idx).

        Args:
            idx: Flat index into all bonds in current file

        Returns:
            chunk_idx: Which chunk this molecule is in
            row_in_chunk: Row index within that chunk
            bond_idx: Which bond within this molecule
        """
        current_sum = 0
        for chunk_idx, row_in_chunk, bond_count in self.valid_molecule_indices:
            if idx < current_sum + bond_count:
                bond_idx = idx - current_sum
                return chunk_idx, row_in_chunk, bond_idx
            current_sum += bond_count

        raise IndexError(f"Index {idx} out of range (total: {self.total_bonds})")

    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample, using RAM cache to avoid repeated disk reads.

        Args:
            idx: Index of bond to load

        Returns:
            PyTorch Geometric Data object
        """
        # Find which molecule and bond
        chunk_idx, row_in_chunk, bond_idx = self._get_molecule_and_bond_index(idx)

        # Try to get chunk from cache
        chunk_to_read = self.chunk_cache.get(self.current_file, chunk_idx)

        if chunk_to_read is None:
            # Cache miss - read from disk
            skip_chunks = chunk_idx
            chunk_to_read = pd.read_csv(
                self.current_file,
                skiprows=range(1, skip_chunks * self.chunk_size + 1),  # Skip header + previous chunks
                nrows=self.chunk_size
            )
            # Add to cache
            self.chunk_cache.put(self.current_file, chunk_idx, chunk_to_read)

        # Get the specific row
        row = chunk_to_read.iloc[row_in_chunk]

        # Process this molecule and bond
        data = self._process_molecule_bond(row, bond_idx)

        return data

    def _process_molecule_bond(self, row, bond_idx: int) -> Data:
        """
        Process a specific molecule and return the bond at bond_idx.

        This is where stereochemistry-aware geometry computation happens.

        Args:
            row: DataFrame row containing molecule data
            bond_idx: Index of bond to process

        Returns:
            PyTorch Geometric Data object with local geometry
        """
        # Parse XYZ and bonds
        coordinates, elements = parse_xyz_string(row['geometry_xyz'])
        bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])

        # Filter valid elements
        valid_indices = [i for i, elem in enumerate(elements) if elem.strip()]
        coordinates = [coordinates[i] for i in valid_indices]
        elements = [elements[i] for i in valid_indices]

        # Create element indices using the provided mapping
        element_indices = []
        for elem in elements:
            elem_idx = self.element_to_idx.get(elem, 0)  # 0 for 'OTHER'
            element_indices.append(elem_idx)
        x = torch.tensor(element_indices, dtype=torch.long)

        pos = torch.tensor(coordinates, dtype=torch.float)

        # Create adjacency graph from dataset bonds
        adjacency = defaultdict(list)
        bond_pairs = []

        # Add bonds from dataset
        for bond in bonds_data:
            if 'i' in bond and 'j' in bond:
                i, j = bond['i'], bond['j']
                if i < len(coordinates) and j < len(coordinates) and i != j:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    dist = torch.norm(pos[i] - pos[j])
                    bond_pairs.append((i, j, dist.item()))

        # Add non-bonded pairs if requested
        if self.include_non_bonded:
            bonded_set = set((min(i,j), max(i,j)) for i,j,_ in bond_pairs)
            non_bonded_pairs = []

            for i in range(len(coordinates)):
                for j in range(i+1, len(coordinates)):
                    if (i,j) not in bonded_set:
                        dist = torch.norm(pos[i] - pos[j])
                        if dist < self.bond_threshold:
                            non_bonded_pairs.append((i, j, dist.item()))

            if non_bonded_pairs:
                n_sample = min(len(bond_pairs) // 3, len(non_bonded_pairs))
                sampled_non_bonded = random.sample(non_bonded_pairs, n_sample)
                bond_pairs.extend(sampled_non_bonded)

        # Get the specific bond
        if bond_idx >= len(bond_pairs):
            raise IndexError(f"Bond index {bond_idx} out of range")

        node_a, node_b, target_distance = bond_pairs[bond_idx]

        # Extract subgraph around both nodes
        subgraph_a = dfs_subgraph(adjacency, node_a, depth=self.subgraph_depth)
        subgraph_b = dfs_subgraph(adjacency, node_b, depth=self.subgraph_depth)
        subgraph_nodes = subgraph_a.union(subgraph_b)
        subgraph_nodes.add(node_a)
        subgraph_nodes.add(node_b)

        # Create node mapping
        node_list = sorted(list(subgraph_nodes))
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_list)}

        # Map target nodes
        new_node_a = node_mapping[node_a]
        new_node_b = node_mapping[node_b]

        # Create subgraph features and positions
        subgraph_x = x[node_list]
        subgraph_pos = pos[node_list]

        # Create subgraph edges
        edge_index = []
        for old_i in node_list:
            for old_j in adjacency[old_i]:
                if old_j in node_mapping:
                    new_i = node_mapping[old_i]
                    new_j = node_mapping[old_j]
                    edge_index.append([new_i, new_j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)

        # **NEW: Compute stereochemistry-aware local geometry**
        # For node A, use A→B bond as reference
        node_a_neighbors = [node_mapping[n] for n in adjacency[node_a] if n in node_mapping]

        if new_node_b in node_a_neighbors:
            # Compute local geometry for node A
            ordered_neighbors_a, geometry_a, frame_a = compute_local_geometry(
                subgraph_pos,
                center_idx=new_node_a,
                neighbor_indices=node_a_neighbors,
                reference_idx=new_node_b,
                augment_rotation=self.training  # Random rotation during training
            )
        else:
            # Fallback: no local geometry
            ordered_neighbors_a = []
            geometry_a = []
            frame_a = None

        # For node B, use B→A bond as reference
        node_b_neighbors = [node_mapping[n] for n in adjacency[node_b] if n in node_mapping]

        if new_node_a in node_b_neighbors:
            # Compute local geometry for node B
            ordered_neighbors_b, geometry_b, frame_b = compute_local_geometry(
                subgraph_pos,
                center_idx=new_node_b,
                neighbor_indices=node_b_neighbors,
                reference_idx=new_node_a,
                augment_rotation=self.training
            )
        else:
            # Fallback: no local geometry
            ordered_neighbors_b = []
            geometry_b = []
            frame_b = None

        # Pack geometry into tensors (r, θ, φ) for up to max_bonds
        geometry_a_padded = []
        for i in range(self.max_bonds):
            if i < len(geometry_a):
                r, theta, phi = geometry_a[i]
                geometry_a_padded.append([r, theta, phi])
            else:
                geometry_a_padded.append([0.0, 0.0, 0.0])  # Padding

        geometry_b_padded = []
        for i in range(self.max_bonds):
            if i < len(geometry_b):
                r, theta, phi = geometry_b[i]
                geometry_b_padded.append([r, theta, phi])
            else:
                geometry_b_padded.append([0.0, 0.0, 0.0])  # Padding

        # Bond masks
        bond_mask_a = torch.zeros(self.max_bonds)
        for i in range(min(len(geometry_a), self.max_bonds)):
            bond_mask_a[i] = 1.0

        bond_mask_b = torch.zeros(self.max_bonds)
        for i in range(min(len(geometry_b), self.max_bonds)):
            bond_mask_b[i] = 1.0

        # Create data object
        data = Data(
            x=subgraph_x,
            edge_index=edge_index,
            pos=subgraph_pos,
            node_a=new_node_a,
            node_b=new_node_b,
            target_distance=torch.tensor([target_distance], dtype=torch.float),
            target_geometry_a=torch.tensor(geometry_a_padded, dtype=torch.float),  # (max_bonds, 3)
            target_geometry_b=torch.tensor(geometry_b_padded, dtype=torch.float),  # (max_bonds, 3)
            bond_mask_a=bond_mask_a,
            bond_mask_b=bond_mask_b
        )

        return data


def analyze_element_frequencies(
    file_paths: List[str],
    min_count: int = 10,
    max_molecules: int = 100000
) -> Dict[str, int]:
    """
    Analyze element frequencies across dataset files.

    Reads files in chunks and stops after max_molecules for efficiency.

    Args:
        file_paths: List of CSV file paths
        min_count: Minimum occurrences for frequent elements
        max_molecules: Maximum molecules to analyze (for speed)

    Returns:
        element_to_idx: Mapping from element to index (0 = 'OTHER')
    """
    element_counts = defaultdict(int)
    files = file_paths if isinstance(file_paths, list) else [file_paths]

    if not files:
        raise ValueError(f"No files provided")

    print(f"Analyzing element frequencies from {len(files)} file(s)...")

    molecules_analyzed = 0

    for file_path in files:
        print(f"  Reading: {file_path}")

        for chunk in pd.read_csv(file_path, chunksize=10000):
            for _, row in chunk.iterrows():
                try:
                    if 'geometry_xyz' not in row or pd.isna(row['geometry_xyz']):
                        continue

                    coordinates, elements = parse_xyz_string(row['geometry_xyz'])
                    if not elements:
                        continue

                    for element in elements:
                        element_counts[element] += 1

                    molecules_analyzed += 1

                    if molecules_analyzed >= max_molecules:
                        break

                except Exception:
                    continue

            if molecules_analyzed >= max_molecules:
                break

        if molecules_analyzed >= max_molecules:
            break

    # Filter by frequency
    frequent_elements = {elem: count for elem, count in element_counts.items() if count >= min_count}
    rare_elements = {elem: count for elem, count in element_counts.items() if count < min_count}

    print(f"\nElement frequency analysis ({molecules_analyzed} molecules):")
    print(f"  Unique elements: {len(element_counts)}")
    print(f"  Frequent (>={min_count}): {len(frequent_elements)}")
    print(f"  Rare (<{min_count}): {len(rare_elements)}")

    # Create mapping (0 reserved for 'OTHER')
    element_to_idx = {'OTHER': 0}
    for i, element in enumerate(sorted(frequent_elements.keys())):
        element_to_idx[element] = i + 1

    print(f"  Vocabulary size: {len(element_to_idx)} (including OTHER)")
    print(f"  Frequent elements: {sorted(frequent_elements.keys())}")

    return element_to_idx
