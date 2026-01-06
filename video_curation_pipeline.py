#!/usr/bin/env python3
"""
Video-Centric Data Curation Pipeline

A comprehensive pipeline for analyzing diversity in driving video clips using:
- VideoMAEv2 for feature extraction
- Leiden Graph Clustering for unsupervised scenario discovery
- Edge case and void detection for identifying missing data

Author: AI Engineer
License: MIT
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# Lazy import for torch (only needed for VideoEmbedder)
if TYPE_CHECKING:
    import torch
    from torch import Tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class VideoEmbedderConfig:
    """Configuration for VideoMAEv2 embedding extraction."""
    model_name: str = "OpenGVLab/VideoMAEv2-Huge"  # VideoMAEv2-Huge (0.6B params)
    num_frames: int = 16
    batch_size: int = 4
    device: str = field(default_factory=_get_device)
    normalize_embeddings: bool = True
    frame_sample_strategy: str = "uniform"  # "uniform" or "random"
    trust_remote_code: bool = True  # Required for OpenGVLab models
    output_dir: str = "output"  # Directory to save embeddings


@dataclass
class ClusteringConfig:
    """Configuration for graph construction and Leiden clustering."""
    k_neighbors: int = 15
    metric: str = "cosine"
    resolution: float = 1.0  # Leiden resolution parameter
    n_iterations: int = -1  # -1 for unlimited iterations
    random_state: int = 42


@dataclass
class AnalysisConfig:
    """Configuration for cluster analysis and void detection."""
    micro_cluster_threshold: int = 3  # Clusters smaller than this are edge cases
    low_density_percentile: float = 10.0  # Bottom X% density = void
    outlier_distance_percentile: float = 95.0  # Top X% distance = outlier


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    cluster_id: int
    size: int
    centroid: NDArray[np.float32]
    density: float
    representative_idx: int
    representative_path: Optional[str] = None
    is_micro_cluster: bool = False
    is_low_density: bool = False


@dataclass
class PipelineResults:
    """Complete results from the curation pipeline."""
    embeddings: NDArray[np.float32]
    cluster_labels: NDArray[np.int32]
    cluster_info: dict[int, ClusterInfo]
    representative_videos: dict[int, str]
    edge_case_indices: list[int]
    void_cluster_ids: list[int]
    captions: dict[int, str] = field(default_factory=dict)


# =============================================================================
# Video Embedding Extraction (VideoMAEv2)
# =============================================================================

class VideoEmbedder:
    """
    Extract dense vector representations from video clips using VideoMAEv2.

    This class handles:
    - Loading the VideoMAEv2-huge model and processor
    - Frame sampling from video files
    - Batch processing for GPU efficiency
    - L2 normalization of embeddings
    """

    def __init__(self, config: Optional[VideoEmbedderConfig] = None):
        """
        Initialize the VideoEmbedder.

        Args:
            config: Configuration for the embedder. Uses defaults if None.
        """
        self.config = config or VideoEmbedderConfig()
        self.device = None  # Lazily initialized
        self.model = None
        self.processor = None
        self._is_loaded = False

    def _ensure_torch(self) -> None:
        """Ensure torch is imported and device is set."""
        if self.device is None:
            import torch
            self.device = torch.device(self.config.device)

    def load_model(self) -> None:
        """Load the VideoMAEv2 model and processor from HuggingFace."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return

        self._ensure_torch()

        try:
            from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig

            logger.info(f"Loading VideoMAEv2 model: {self.config.model_name}")

            # OpenGVLab/VideoMAEv2-Huge requires trust_remote_code=True
            config = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )

            self.processor = VideoMAEImageProcessor.from_pretrained(
                self.config.model_name
            )

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                config=config,
                trust_remote_code=self.config.trust_remote_code
            ).to(self.device)
            self.model.eval()

            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

        except ImportError as e:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    @staticmethod
    def extract_video_name_from_frame_dir(frame_dir: Union[str, Path]) -> str:
        """
        Extract original video name from frame directory path.

        Expected path structure:
        ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/*.jpg

        The video name is extracted from the parent directory structure:
        N[숫자7자리]-[YYMMDDhhmmss].mp4

        Args:
            frame_dir: Path to the frame directory (CMR_GT_Frame folder or its parent)

        Returns:
            Original video filename (e.g., "N1234567-231215120000.mp4")
        """
        import re

        path = Path(frame_dir)

        # Navigate up to find the directory matching N[7digits]-[timestamp] pattern
        # Pattern: N followed by 7 digits, dash, then 12 digits (YYMMDDhhmmss)
        pattern = re.compile(r'^N\d{7}-\d{12}$')

        current = path
        for _ in range(10):  # Limit search depth
            if pattern.match(current.name):
                return f"{current.name}.mp4"
            if current.parent == current:
                break
            current = current.parent

        # Fallback: use the frame directory name
        logger.warning(f"Could not extract video name from {frame_dir}, using directory name")
        return f"{path.name}.mp4"

    def _sample_frames_from_directory(
        self,
        frame_dir: Union[str, Path]
    ) -> NDArray[np.uint8]:
        """
        Sample frames from a directory of JPG files.

        Args:
            frame_dir: Path to directory containing frame JPG files.

        Returns:
            Array of shape (num_frames, H, W, C) with uint8 values.
        """
        from PIL import Image

        frame_dir = Path(frame_dir)

        # Get all jpg files sorted by name
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            # Try uppercase extension
            frame_files = sorted(frame_dir.glob("*.JPG"))

        if not frame_files:
            raise ValueError(f"No JPG files found in {frame_dir}")

        total_frames = len(frame_files)

        # Sample frame indices
        if total_frames < self.config.num_frames:
            # Repeat frames if not enough
            indices = np.linspace(
                0, total_frames - 1, self.config.num_frames, dtype=int
            )
        elif self.config.frame_sample_strategy == "uniform":
            indices = np.linspace(
                0, total_frames - 1, self.config.num_frames, dtype=int
            )
        else:  # random sampling
            indices = np.sort(
                np.random.choice(
                    total_frames, self.config.num_frames, replace=False
                )
            )

        # Load selected frames
        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert("RGB")
            frames.append(np.array(img))

        return np.stack(frames)

    def _sample_frames_decord(
        self,
        video_path: Union[str, Path]
    ) -> NDArray[np.uint8]:
        """
        Sample frames from video using decord (preferred for performance).

        Args:
            video_path: Path to the video file.

        Returns:
            Array of shape (num_frames, H, W, C) with uint8 values.
        """
        try:
            import decord
            decord.bridge.set_bridge("numpy")

            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)

            if total_frames < self.config.num_frames:
                # Repeat frames if video is too short
                indices = np.linspace(
                    0, total_frames - 1, self.config.num_frames, dtype=int
                )
            elif self.config.frame_sample_strategy == "uniform":
                indices = np.linspace(
                    0, total_frames - 1, self.config.num_frames, dtype=int
                )
            else:  # random sampling
                indices = np.sort(
                    np.random.choice(
                        total_frames, self.config.num_frames, replace=False
                    )
                )

            frames = vr.get_batch(indices).asnumpy()
            return frames

        except ImportError:
            logger.warning("decord not available, falling back to PyAV")
            return self._sample_frames_pyav(video_path)

    def _sample_frames_pyav(
        self,
        video_path: Union[str, Path]
    ) -> NDArray[np.uint8]:
        """
        Sample frames from video using PyAV (fallback).

        Args:
            video_path: Path to the video file.

        Returns:
            Array of shape (num_frames, H, W, C) with uint8 values.
        """
        import av

        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Get total frames
        total_frames = stream.frames
        if total_frames == 0:
            # Estimate from duration
            total_frames = int(
                stream.duration * stream.time_base * stream.average_rate
            )

        # Calculate frame indices to sample
        if self.config.frame_sample_strategy == "uniform":
            indices = set(
                np.linspace(
                    0, max(total_frames - 1, 0),
                    self.config.num_frames,
                    dtype=int
                )
            )
        else:
            indices = set(
                np.sort(
                    np.random.choice(
                        total_frames,
                        min(self.config.num_frames, total_frames),
                        replace=False
                    )
                )
            )

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)
            if len(frames) >= self.config.num_frames:
                break

        container.close()

        # Pad if necessary
        while len(frames) < self.config.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return np.stack(frames[:self.config.num_frames])

    def _process_frames(
        self,
        frames: NDArray[np.uint8]
    ) -> Tensor:
        """
        Process frames through the VideoMAE processor.

        Args:
            frames: Array of shape (num_frames, H, W, C).

        Returns:
            Processed tensor ready for model input.
            Shape: (B, C, T, H, W) for OpenGVLab/VideoMAEv2-Huge
        """
        # Convert frames from (T, H, W, C) to (T, C, H, W) for OpenGVLab model
        # The processor expects frames in (T, C, H, W) format
        frames_transposed = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)

        # Convert to list for processor
        frame_list = [frames_transposed[i] for i in range(frames_transposed.shape[0])]

        inputs = self.processor(
            frame_list,
            return_tensors="pt"
        )

        # Permute from (B, T, C, H, W) -> (B, C, T, H, W) for OpenGVLab/VideoMAEv2-Huge
        pixel_values = inputs["pixel_values"].permute(0, 2, 1, 3, 4)

        return pixel_values.to(self.device)

    def extract_embedding(
        self,
        video_path: Union[str, Path]
    ) -> NDArray[np.float32]:
        """
        Extract embedding for a single video.

        Args:
            video_path: Path to the video file.

        Returns:
            L2-normalized embedding vector.

        Raises:
            RuntimeError: If video cannot be processed.
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        try:
            with torch.no_grad():
                # Sample frames
                frames = self._sample_frames_decord(video_path)

                # Process and get model output
                pixel_values = self._process_frames(frames)
                outputs = self.model(pixel_values)

                # Use mean pooling over sequence dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

                # Normalize if configured
                if self.config.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                return embedding.cpu().numpy().astype(np.float32)

        except Exception as e:
            raise RuntimeError(
                f"Failed to process video {video_path}: {e}"
            ) from e

    def extract_embeddings_batch(
        self,
        video_paths: list[Union[str, Path]],
        show_progress: bool = True
    ) -> tuple[NDArray[np.float32], list[int]]:
        """
        Extract embeddings for multiple videos with batch processing.

        Args:
            video_paths: List of paths to video files.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (embeddings array, list of failed indices).
        """
        if not self._is_loaded:
            self.load_model()

        embeddings = []
        failed_indices = []

        # Optional progress bar
        iterator = video_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(video_paths, desc="Extracting embeddings")
            except ImportError:
                pass

        batch_frames = []
        batch_indices = []

        for idx, video_path in enumerate(iterator):
            try:
                frames = self._sample_frames_decord(video_path)
                batch_frames.append(frames)
                batch_indices.append(idx)

                # Process batch when full
                if len(batch_frames) >= self.config.batch_size:
                    batch_embeddings = self._process_batch(batch_frames)
                    embeddings.extend(batch_embeddings)
                    batch_frames = []
                    batch_indices = []

            except Exception as e:
                logger.warning(f"Failed to process {video_path}: {e}")
                failed_indices.append(idx)

        # Process remaining batch
        if batch_frames:
            batch_embeddings = self._process_batch(batch_frames)
            embeddings.extend(batch_embeddings)

        if not embeddings:
            raise RuntimeError("No videos could be processed successfully")

        return np.stack(embeddings), failed_indices

    def extract_embedding_from_frame_dir(
        self,
        frame_dir: Union[str, Path],
        save_embedding: bool = True
    ) -> tuple[NDArray[np.float32], str]:
        """
        Extract embedding from a directory of frame JPG files.

        Args:
            frame_dir: Path to directory containing frame JPG files.
            save_embedding: Whether to save the embedding to output directory.

        Returns:
            Tuple of (embedding vector, video name).
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        frame_dir = Path(frame_dir)
        video_name = self.extract_video_name_from_frame_dir(frame_dir)

        try:
            with torch.no_grad():
                # Sample frames from directory
                frames = self._sample_frames_from_directory(frame_dir)

                # Process and get model output
                pixel_values = self._process_frames(frames)
                outputs = self.model(pixel_values)

                # Use mean pooling over sequence dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

                # Normalize if configured
                if self.config.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                embedding_np = embedding.cpu().numpy().astype(np.float32)

                # Save embedding if requested
                if save_embedding:
                    self._save_embedding(embedding_np, video_name)

                return embedding_np, video_name

        except Exception as e:
            raise RuntimeError(
                f"Failed to process frame directory {frame_dir}: {e}"
            ) from e

    def _save_embedding(
        self,
        embedding: NDArray[np.float32],
        video_name: str
    ) -> Path:
        """
        Save embedding to output directory with video name.

        Args:
            embedding: Embedding vector to save.
            video_name: Original video filename (e.g., "N1234567-231215120000.mp4")

        Returns:
            Path to saved embedding file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Replace .mp4 with .npy
        embedding_filename = video_name.replace(".mp4", ".npy")
        output_path = output_dir / embedding_filename

        np.save(output_path, embedding)
        logger.debug(f"Saved embedding to {output_path}")

        return output_path

    def extract_embeddings_from_frame_dirs(
        self,
        frame_dirs: list[Union[str, Path]],
        save_embeddings: bool = True,
        show_progress: bool = True
    ) -> tuple[NDArray[np.float32], list[str], list[int]]:
        """
        Extract embeddings from multiple frame directories.

        Args:
            frame_dirs: List of paths to frame directories.
            save_embeddings: Whether to save embeddings to output directory.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (embeddings array, video names, failed indices).
        """
        if not self._is_loaded:
            self.load_model()

        embeddings = []
        video_names = []
        failed_indices = []

        # Ensure output directory exists
        if save_embeddings:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Optional progress bar
        iterator = enumerate(frame_dirs)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Extracting embeddings from frames")
            except ImportError:
                iterator = enumerate(frame_dirs)

        for idx, frame_dir in iterator:
            try:
                embedding, video_name = self.extract_embedding_from_frame_dir(
                    frame_dir, save_embedding=save_embeddings
                )
                embeddings.append(embedding)
                video_names.append(video_name)

            except Exception as e:
                logger.warning(f"Failed to process {frame_dir}: {e}")
                failed_indices.append(idx)

        if not embeddings:
            raise RuntimeError("No frame directories could be processed successfully")

        return np.stack(embeddings), video_names, failed_indices

    @staticmethod
    def find_frame_directories(
        base_dir: Union[str, Path],
        pattern: str = "**/CMR_GT_Frame"
    ) -> list[Path]:
        """
        Find all frame directories matching the expected structure.

        Expected structure:
        ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/

        Args:
            base_dir: Base directory to search from.
            pattern: Glob pattern to match frame directories.

        Returns:
            List of paths to frame directories.
        """
        base_dir = Path(base_dir)
        frame_dirs = sorted(base_dir.glob(pattern))
        logger.info(f"Found {len(frame_dirs)} frame directories in {base_dir}")
        return frame_dirs

    def _process_batch(
        self,
        batch_frames: list[NDArray[np.uint8]]
    ) -> list[NDArray[np.float32]]:
        """
        Process a batch of frame arrays through the model.

        Args:
            batch_frames: List of frame arrays.

        Returns:
            List of embedding vectors.
        """
        import torch

        embeddings = []

        with torch.no_grad():
            for frames in batch_frames:
                pixel_values = self._process_frames(frames)
                outputs = self.model(pixel_values)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

                if self.config.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                embeddings.append(embedding.cpu().numpy().astype(np.float32))

        return embeddings


# =============================================================================
# Graph Construction & Leiden Clustering
# =============================================================================

class GraphClusterer:
    """
    Construct k-NN graph and perform Leiden community detection.

    This approach captures manifold structure better than K-Means for
    complex driving scenarios with non-convex cluster shapes.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the GraphClusterer.

        Args:
            config: Configuration for clustering. Uses defaults if None.
        """
        self.config = config or ClusteringConfig()
        self.knn_graph_ = None
        self.adjacency_matrix_ = None

    def _build_knn_graph(
        self,
        embeddings: NDArray[np.float32]
    ) -> Any:
        """
        Build approximate k-NN graph from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).

        Returns:
            Sparse adjacency matrix.
        """
        n_samples = embeddings.shape[0]
        k = min(self.config.k_neighbors, n_samples - 1)

        logger.info(f"Building k-NN graph with k={k}")

        # Try pynndescent for approximate NN (faster for large datasets)
        try:
            from pynndescent import NNDescent

            index = NNDescent(
                embeddings,
                metric=self.config.metric,
                n_neighbors=k + 1,  # +1 because it includes self
                random_state=self.config.random_state
            )
            indices, distances = index.neighbor_graph

            # Remove self-connections
            indices = indices[:, 1:]
            distances = distances[:, 1:]

        except ImportError:
            logger.info("pynndescent not available, using sklearn")
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(
                n_neighbors=k,
                metric=self.config.metric,
                algorithm="auto"
            )
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)

        # Build sparse adjacency matrix
        from scipy import sparse

        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()

        # Use similarity (1 - distance for cosine)
        if self.config.metric == "cosine":
            similarities = 1 - distances.flatten()
        else:
            # For euclidean, use Gaussian kernel
            sigma = np.median(distances)
            similarities = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))

        # Create symmetric adjacency matrix
        adjacency = sparse.csr_matrix(
            (similarities, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        # Make symmetric
        adjacency = adjacency + adjacency.T
        adjacency.data = np.minimum(adjacency.data, 1.0)

        self.adjacency_matrix_ = adjacency
        return adjacency

    def _leiden_clustering(
        self,
        adjacency: Any
    ) -> NDArray[np.int32]:
        """
        Apply Leiden algorithm for community detection.

        Args:
            adjacency: Sparse adjacency matrix.

        Returns:
            Array of cluster labels.
        """
        logger.info("Running Leiden clustering...")

        # Try leidenalg (most reliable)
        try:
            import igraph as ig
            import leidenalg

            # Convert to igraph
            sources, targets = adjacency.nonzero()
            weights = adjacency[sources, targets].A1

            g = ig.Graph(
                n=adjacency.shape[0],
                edges=list(zip(sources.tolist(), targets.tolist())),
                directed=False
            )
            g.es["weight"] = weights.tolist()

            # Run Leiden
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=self.config.resolution,
                n_iterations=self.config.n_iterations,
                seed=self.config.random_state
            )

            return np.array(partition.membership, dtype=np.int32)

        except ImportError:
            logger.warning("leidenalg not available, trying cdlib")

        # Fallback to cdlib
        try:
            import networkx as nx
            from cdlib import algorithms

            G = nx.from_scipy_sparse_array(adjacency)

            # cdlib Leiden
            communities = algorithms.leiden(
                G,
                weights="weight",
                resolution_parameter=self.config.resolution
            )

            # Convert to labels
            labels = np.zeros(adjacency.shape[0], dtype=np.int32)
            for cluster_id, members in enumerate(communities.communities):
                for member in members:
                    labels[member] = cluster_id

            return labels

        except ImportError:
            logger.warning("cdlib not available, falling back to Louvain")

        # Final fallback to sklearn spectral clustering
        from sklearn.cluster import SpectralClustering

        n_clusters = max(2, int(np.sqrt(adjacency.shape[0] / 2)))

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=self.config.random_state
        )

        return clustering.fit_predict(adjacency.toarray()).astype(np.int32)

    def fit_predict(
        self,
        embeddings: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """
        Build graph and perform clustering.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).

        Returns:
            Array of cluster labels.
        """
        adjacency = self._build_knn_graph(embeddings)
        labels = self._leiden_clustering(adjacency)

        n_clusters = len(np.unique(labels))
        logger.info(f"Found {n_clusters} clusters")

        return labels


# =============================================================================
# Cluster Analysis & Edge Case Discovery
# =============================================================================

class ClusterAnalyzer:
    """
    Analyze clusters to find representatives, edge cases, and voids.

    - Computes cluster centroids and finds representative samples
    - Identifies micro-clusters (edge cases)
    - Detects sparse regions (voids/under-represented scenarios)
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the ClusterAnalyzer.

        Args:
            config: Configuration for analysis. Uses defaults if None.
        """
        self.config = config or AnalysisConfig()

    def _compute_cluster_centroids(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32]
    ) -> dict[int, NDArray[np.float32]]:
        """
        Compute centroid for each cluster.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.

        Returns:
            Dictionary mapping cluster_id to centroid vector.
        """
        centroids = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            # Normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[int(label)] = centroid

        return centroids

    def _compute_cluster_density(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]]
    ) -> dict[int, float]:
        """
        Compute density for each cluster (inverse of avg distance to centroid).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.

        Returns:
            Dictionary mapping cluster_id to density score.
        """
        densities = {}

        for cluster_id, centroid in centroids.items():
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]

            # Compute distances to centroid
            distances = np.linalg.norm(
                cluster_embeddings - centroid, axis=1
            )

            avg_distance = distances.mean() if len(distances) > 0 else 1.0
            # Density is inverse of average distance
            densities[cluster_id] = 1.0 / (avg_distance + 1e-8)

        return densities

    def _find_representatives(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]],
        video_paths: Optional[list[str]] = None
    ) -> dict[int, tuple[int, Optional[str]]]:
        """
        Find the most representative sample for each cluster.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.
            video_paths: Optional list of video paths.

        Returns:
            Dictionary mapping cluster_id to (sample_idx, video_path).
        """
        representatives = {}

        for cluster_id, centroid in centroids.items():
            mask = labels == cluster_id
            indices = np.where(mask)[0]
            cluster_embeddings = embeddings[mask]

            # Find closest to centroid
            distances = np.linalg.norm(
                cluster_embeddings - centroid, axis=1
            )
            closest_idx = indices[np.argmin(distances)]

            path = video_paths[closest_idx] if video_paths else None
            representatives[cluster_id] = (int(closest_idx), path)

        return representatives

    def _identify_edge_cases(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]],
        cluster_sizes: dict[int, int]
    ) -> list[int]:
        """
        Identify edge case samples (outliers and micro-cluster members).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.
            cluster_sizes: Dictionary of cluster sizes.

        Returns:
            List of edge case sample indices.
        """
        edge_cases = set()

        # 1. Members of micro-clusters
        micro_clusters = {
            cid for cid, size in cluster_sizes.items()
            if size < self.config.micro_cluster_threshold
        }

        for idx, label in enumerate(labels):
            if label in micro_clusters:
                edge_cases.add(idx)

        # 2. Outliers: samples far from their cluster centroid
        distances_to_centroid = []
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            centroid = centroids[int(label)]
            dist = np.linalg.norm(embedding - centroid)
            distances_to_centroid.append((idx, dist))

        # Find outliers based on distance percentile
        all_distances = [d for _, d in distances_to_centroid]
        threshold = np.percentile(
            all_distances,
            self.config.outlier_distance_percentile
        )

        for idx, dist in distances_to_centroid:
            if dist > threshold:
                edge_cases.add(idx)

        return sorted(list(edge_cases))

    def _identify_voids(
        self,
        densities: dict[int, float],
        cluster_sizes: dict[int, int]
    ) -> list[int]:
        """
        Identify void clusters (sparse, under-represented scenarios).

        Args:
            densities: Dictionary of cluster densities.
            cluster_sizes: Dictionary of cluster sizes.

        Returns:
            List of void cluster IDs.
        """
        voids = []

        density_values = list(densities.values())
        threshold = np.percentile(
            density_values,
            self.config.low_density_percentile
        )

        for cluster_id, density in densities.items():
            # Low density AND not a micro-cluster
            # (micro-clusters are handled separately as edge cases)
            if (density < threshold and
                cluster_sizes[cluster_id] >= self.config.micro_cluster_threshold):
                voids.append(cluster_id)

        return voids

    def analyze(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        video_paths: Optional[list[str]] = None
    ) -> tuple[dict[int, ClusterInfo], list[int], list[int]]:
        """
        Perform complete cluster analysis.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            video_paths: Optional list of video paths.

        Returns:
            Tuple of (cluster_info dict, edge_case_indices, void_cluster_ids).
        """
        logger.info("Analyzing clusters...")

        # Compute cluster statistics
        unique_labels = np.unique(labels)
        cluster_sizes = {
            int(label): int(np.sum(labels == label))
            for label in unique_labels
        }

        centroids = self._compute_cluster_centroids(embeddings, labels)
        densities = self._compute_cluster_density(embeddings, labels, centroids)
        representatives = self._find_representatives(
            embeddings, labels, centroids, video_paths
        )

        # Identify edge cases and voids
        edge_cases = self._identify_edge_cases(
            embeddings, labels, centroids, cluster_sizes
        )
        voids = self._identify_voids(densities, cluster_sizes)

        # Build cluster info
        micro_clusters = {
            cid for cid, size in cluster_sizes.items()
            if size < self.config.micro_cluster_threshold
        }

        cluster_info = {}
        for cluster_id in unique_labels:
            cid = int(cluster_id)
            rep_idx, rep_path = representatives[cid]

            cluster_info[cid] = ClusterInfo(
                cluster_id=cid,
                size=cluster_sizes[cid],
                centroid=centroids[cid],
                density=densities[cid],
                representative_idx=rep_idx,
                representative_path=rep_path,
                is_micro_cluster=cid in micro_clusters,
                is_low_density=cid in voids
            )

        logger.info(f"Found {len(edge_cases)} edge cases")
        logger.info(f"Found {len(voids)} void clusters")

        return cluster_info, edge_cases, voids


# =============================================================================
# Captioning Interface (Mock VLM Integration)
# =============================================================================

class CaptioningInterface(ABC):
    """Abstract base class for video captioning interfaces."""

    @abstractmethod
    def caption_video(
        self,
        video_path: str,
        cluster_id: int
    ) -> str:
        """Generate caption for a video."""
        pass

    def generate_cluster_captions(
        self,
        representative_videos: dict[int, str]
    ) -> dict[int, str]:
        """
        Generate captions for all representative videos.

        Args:
            representative_videos: Mapping of cluster_id to video path.

        Returns:
            Mapping of cluster_id to caption string.
        """
        captions = {}
        for cluster_id, video_path in representative_videos.items():
            captions[cluster_id] = self.caption_video(video_path, cluster_id)
        return captions


class MockCaptioningInterface(CaptioningInterface):
    """
    Mock captioning interface for demonstration.

    In production, this would be replaced with actual VLM API calls
    (e.g., GPT-4V, Claude Vision, or specialized video understanding models).
    """

    # Simulated scenario types for demo
    SCENARIO_TYPES = [
        "Rainy night on highway",
        "Dense urban traffic with pedestrians",
        "Construction zone with lane closures",
        "Parking lot navigation",
        "Highway merge with fast traffic",
        "Foggy morning on rural road",
        "School zone during dismissal",
        "Roundabout navigation",
        "Emergency vehicle approaching",
        "Cyclist in bike lane",
        "Deer crossing warning area",
        "Tunnel entrance with lighting change",
        "Snow-covered residential street",
        "Loading dock area",
        "Multi-lane intersection",
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize mock captioning interface.

        Args:
            seed: Random seed for reproducible captions.
        """
        self.rng = np.random.default_rng(seed)

    def caption_video(
        self,
        video_path: str,
        cluster_id: int
    ) -> str:
        """
        Generate a mock caption for a video.

        In production, replace this with:

        ```python
        # Example GPT-4V API call (pseudocode)
        from openai import OpenAI

        client = OpenAI()

        # Extract key frames from video
        frames = extract_key_frames(video_path, n_frames=4)

        # Encode frames as base64
        images = [encode_image(frame) for frame in frames]

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this driving scenario..."},
                    *[{"type": "image_url", "image_url": img} for img in images]
                ]
            }]
        )
        return response.choices[0].message.content
        ```

        Args:
            video_path: Path to the video file.
            cluster_id: ID of the cluster this video represents.

        Returns:
            Caption string describing the scenario.
        """
        # Mock: Select scenario based on cluster_id for consistency
        scenario_idx = cluster_id % len(self.SCENARIO_TYPES)
        scenario = self.SCENARIO_TYPES[scenario_idx]

        return f"Cluster {cluster_id}: {scenario}"


# =============================================================================
# Main Pipeline Orchestration
# =============================================================================

class VideoCurationPipeline:
    """
    Main orchestration class for the video curation pipeline.

    Combines all components:
    - VideoEmbedder for feature extraction
    - GraphClusterer for Leiden clustering
    - ClusterAnalyzer for analysis and void detection
    - CaptioningInterface for scenario descriptions
    """

    def __init__(
        self,
        embedder_config: Optional[VideoEmbedderConfig] = None,
        clustering_config: Optional[ClusteringConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        captioning_interface: Optional[CaptioningInterface] = None
    ):
        """
        Initialize the pipeline.

        Args:
            embedder_config: Configuration for video embedding.
            clustering_config: Configuration for graph clustering.
            analysis_config: Configuration for cluster analysis.
            captioning_interface: Interface for video captioning.
        """
        self.embedder = VideoEmbedder(embedder_config)
        self.clusterer = GraphClusterer(clustering_config)
        self.analyzer = ClusterAnalyzer(analysis_config)
        self.captioner = captioning_interface or MockCaptioningInterface()

    def run(
        self,
        video_paths: list[str],
        generate_captions: bool = True
    ) -> PipelineResults:
        """
        Run the complete curation pipeline on a set of videos.

        Args:
            video_paths: List of paths to video files.
            generate_captions: Whether to generate cluster captions.

        Returns:
            Complete pipeline results.
        """
        logger.info(f"Starting pipeline with {len(video_paths)} videos")

        # Step 1: Extract embeddings
        logger.info("Step 1: Extracting video embeddings...")
        embeddings, failed_indices = self.embedder.extract_embeddings_batch(
            video_paths
        )

        # Filter out failed videos
        valid_paths = [
            p for i, p in enumerate(video_paths)
            if i not in failed_indices
        ]

        if len(failed_indices) > 0:
            logger.warning(f"{len(failed_indices)} videos failed to process")

        # Step 2: Graph clustering
        logger.info("Step 2: Performing Leiden clustering...")
        labels = self.clusterer.fit_predict(embeddings)

        # Step 3: Cluster analysis
        logger.info("Step 3: Analyzing clusters...")
        cluster_info, edge_cases, voids = self.analyzer.analyze(
            embeddings, labels, valid_paths
        )

        # Build representative videos mapping
        representative_videos = {
            cid: info.representative_path
            for cid, info in cluster_info.items()
            if info.representative_path is not None
        }

        # Step 4: Generate captions
        captions = {}
        if generate_captions and representative_videos:
            logger.info("Step 4: Generating cluster captions...")
            captions = self.captioner.generate_cluster_captions(
                representative_videos
            )

        logger.info("Pipeline complete!")

        return PipelineResults(
            embeddings=embeddings,
            cluster_labels=labels,
            cluster_info=cluster_info,
            representative_videos=representative_videos,
            edge_case_indices=edge_cases,
            void_cluster_ids=voids,
            captions=captions
        )

    def run_from_embeddings(
        self,
        embeddings: NDArray[np.float32],
        video_paths: Optional[list[str]] = None,
        generate_captions: bool = True
    ) -> PipelineResults:
        """
        Run pipeline from pre-computed embeddings (for demo/testing).

        Args:
            embeddings: Pre-computed embedding vectors.
            video_paths: Optional list of video paths.
            generate_captions: Whether to generate cluster captions.

        Returns:
            Complete pipeline results.
        """
        logger.info(f"Running pipeline with {len(embeddings)} embeddings")

        # Step 1: Graph clustering
        logger.info("Step 1: Performing Leiden clustering...")
        labels = self.clusterer.fit_predict(embeddings)

        # Step 2: Cluster analysis
        logger.info("Step 2: Analyzing clusters...")
        cluster_info, edge_cases, voids = self.analyzer.analyze(
            embeddings, labels, video_paths
        )

        # Build representative videos mapping
        representative_videos = {}
        for cid, info in cluster_info.items():
            if info.representative_path:
                representative_videos[cid] = info.representative_path
            else:
                # Use dummy path for demo
                representative_videos[cid] = f"video_{info.representative_idx}.mp4"

        # Step 3: Generate captions
        captions = {}
        if generate_captions:
            logger.info("Step 3: Generating cluster captions...")
            captions = self.captioner.generate_cluster_captions(
                representative_videos
            )

        logger.info("Pipeline complete!")

        return PipelineResults(
            embeddings=embeddings,
            cluster_labels=labels,
            cluster_info=cluster_info,
            representative_videos=representative_videos,
            edge_case_indices=edge_cases,
            void_cluster_ids=voids,
            captions=captions
        )


# =============================================================================
# Demo Utilities
# =============================================================================

def generate_synthetic_embeddings(
    n_samples: int = 500,
    embedding_dim: int = 1280,  # VideoMAE-huge output dim
    n_clusters: int = 10,
    noise_level: float = 0.3,
    seed: int = 42
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Generate synthetic embeddings for demonstration.

    Creates embeddings with known cluster structure including:
    - Dense clusters (well-represented scenarios)
    - Sparse clusters (voids)
    - Micro-clusters (edge cases)
    - Outliers

    Args:
        n_samples: Total number of samples to generate.
        embedding_dim: Dimension of embedding vectors.
        n_clusters: Number of clusters to create.
        noise_level: Amount of noise to add.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (embeddings, ground_truth_labels).
    """
    rng = np.random.default_rng(seed)

    embeddings = []
    labels = []

    # Generate cluster centers on unit hypersphere
    centers = rng.standard_normal((n_clusters, embedding_dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Assign different sizes to clusters
    # Most clusters are medium, some sparse, some micro
    cluster_sizes = []
    remaining = n_samples

    for i in range(n_clusters):
        if i < 2:  # 2 micro-clusters
            size = rng.integers(1, 3)
        elif i < 4:  # 2 sparse clusters
            size = rng.integers(10, 20)
        else:  # Regular clusters
            size = max(1, remaining // (n_clusters - i))

        size = min(size, remaining)
        cluster_sizes.append(size)
        remaining -= size

    # Distribute any remaining samples
    if remaining > 0:
        cluster_sizes[-1] += remaining

    # Generate samples for each cluster
    for cluster_id, (center, size) in enumerate(zip(centers, cluster_sizes)):
        # Add noise around center
        noise = rng.standard_normal((size, embedding_dim)) * noise_level
        cluster_samples = center + noise

        # Normalize to unit sphere
        cluster_samples = cluster_samples / np.linalg.norm(
            cluster_samples, axis=1, keepdims=True
        )

        embeddings.append(cluster_samples)
        labels.extend([cluster_id] * size)

    # Add some outliers
    n_outliers = max(5, n_samples // 50)
    outliers = rng.standard_normal((n_outliers, embedding_dim))
    outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings.append(outliers)
    labels.extend([n_clusters] * n_outliers)  # New cluster for outliers

    embeddings = np.vstack(embeddings).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)

    return embeddings, labels


def print_results_summary(results: PipelineResults) -> None:
    """Print a human-readable summary of pipeline results."""
    print("\n" + "=" * 60)
    print("VIDEO CURATION PIPELINE RESULTS")
    print("=" * 60)

    print(f"\nTotal videos processed: {len(results.embeddings)}")
    print(f"Number of clusters: {len(results.cluster_info)}")
    print(f"Edge cases identified: {len(results.edge_case_indices)}")
    print(f"Void clusters (under-represented): {len(results.void_cluster_ids)}")

    print("\n" + "-" * 60)
    print("CLUSTER DETAILS")
    print("-" * 60)

    for cid, info in sorted(results.cluster_info.items()):
        status = []
        if info.is_micro_cluster:
            status.append("MICRO-CLUSTER")
        if info.is_low_density:
            status.append("LOW-DENSITY")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"\nCluster {cid}{status_str}:")
        print(f"  Size: {info.size} videos")
        print(f"  Density: {info.density:.4f}")
        print(f"  Representative: {info.representative_path or f'idx={info.representative_idx}'}")

        if cid in results.captions:
            print(f"  Caption: {results.captions[cid]}")

    if results.edge_case_indices:
        print("\n" + "-" * 60)
        print("EDGE CASES (Outliers & Micro-cluster members)")
        print("-" * 60)
        print(f"Indices: {results.edge_case_indices[:20]}")
        if len(results.edge_case_indices) > 20:
            print(f"  ... and {len(results.edge_case_indices) - 20} more")

    if results.void_cluster_ids:
        print("\n" + "-" * 60)
        print("VOID CLUSTERS (Under-represented scenarios)")
        print("-" * 60)
        print(f"Cluster IDs: {results.void_cluster_ids}")
        print("These clusters may indicate missing training data!")

    print("\n" + "=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video-Centric Data Curation Pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic embeddings"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--frame-dir",
        type=str,
        default=None,
        help="Base directory containing frame directories (trainlake structure)"
    )
    parser.add_argument(
        "--frame-pattern",
        type=str,
        default="**/CMR_GT_Frame",
        help="Glob pattern to find frame directories (default: **/CMR_GT_Frame)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save embeddings (default: output)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of synthetic samples for demo mode"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for k-NN graph"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only extract embeddings, skip clustering analysis"
    )

    args = parser.parse_args()

    # Configuration
    embedder_config = VideoEmbedderConfig(
        output_dir=args.output_dir
    )

    clustering_config = ClusteringConfig(
        k_neighbors=args.k_neighbors,
        resolution=args.resolution
    )

    analysis_config = AnalysisConfig(
        micro_cluster_threshold=3,
        low_density_percentile=15.0,
        outlier_distance_percentile=95.0
    )

    # Create pipeline
    pipeline = VideoCurationPipeline(
        embedder_config=embedder_config,
        clustering_config=clustering_config,
        analysis_config=analysis_config
    )

    if args.frame_dir is not None:
        # Frame directory mode: process pre-extracted frames
        # Expected structure: ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/*.jpg
        from pathlib import Path

        print(f"Running in FRAME DIRECTORY MODE...")
        print(f"Base directory: {args.frame_dir}")
        print(f"Frame pattern: {args.frame_pattern}")
        print(f"Output directory: {args.output_dir}")

        # Find all frame directories
        frame_dirs = VideoEmbedder.find_frame_directories(
            args.frame_dir,
            pattern=args.frame_pattern
        )

        if not frame_dirs:
            print(f"No frame directories found matching pattern '{args.frame_pattern}' in {args.frame_dir}")
            exit(1)

        print(f"Found {len(frame_dirs)} frame directories to process")

        # Extract embeddings
        embeddings, video_names, failed_indices = pipeline.embedder.extract_embeddings_from_frame_dirs(
            frame_dirs,
            save_embeddings=True,
            show_progress=True
        )

        print(f"\nEmbedding extraction complete!")
        print(f"  Successful: {len(video_names)}")
        print(f"  Failed: {len(failed_indices)}")
        print(f"  Output directory: {args.output_dir}")

        if not args.skip_clustering and len(embeddings) > 10:
            # Run clustering analysis
            print("\nRunning clustering analysis...")
            results = pipeline.run_from_embeddings(
                embeddings=embeddings,
                video_paths=video_names,
                generate_captions=True
            )

            # Print results
            print_results_summary(results)
        elif args.skip_clustering:
            print("\nSkipping clustering analysis (--skip-clustering flag set)")
        else:
            print("\nSkipping clustering analysis (not enough samples)")

    elif args.demo or args.video_dir is None:
        # Demo mode: use synthetic embeddings
        print("Running in DEMO MODE with synthetic embeddings...")
        print(f"Generating {args.n_samples} synthetic video embeddings...")

        embeddings, ground_truth = generate_synthetic_embeddings(
            n_samples=args.n_samples,
            n_clusters=12
        )

        print(f"Ground truth clusters: {len(np.unique(ground_truth))}")

        # Run pipeline
        results = pipeline.run_from_embeddings(
            embeddings=embeddings,
            generate_captions=True
        )

        # Print results
        print_results_summary(results)

        # Evaluate clustering quality (since we have ground truth)
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(ground_truth, results.cluster_labels)
        nmi = normalized_mutual_info_score(ground_truth, results.cluster_labels)

        print("\nCLUSTERING QUALITY (vs. ground truth):")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Info: {nmi:.4f}")

    else:
        # Real mode: process actual video files
        from pathlib import Path

        video_dir = Path(args.video_dir)
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        video_paths = [
            str(p) for p in video_dir.rglob("*")
            if p.suffix.lower() in video_extensions
        ]

        if not video_paths:
            print(f"No video files found in {video_dir}")
            exit(1)

        print(f"Found {len(video_paths)} videos in {video_dir}")

        # Run pipeline
        results = pipeline.run(
            video_paths=video_paths,
            generate_captions=True
        )

        # Print results
        print_results_summary(results)
