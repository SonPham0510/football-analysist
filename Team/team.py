import logging
from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.

        batch_size (int): The size of each batch.
    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    PCA for dimensionality reduction, and KMeans for clustering.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32, n_cluster: int = 2):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size
        logger.info(
            f"Initializing TeamClassifier on {self.device} with batch size {self.batch_size}"
        )
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(
            device
        )
        self.features_model.eval()  # Set the model to evaluation mode

        self.n_cluster = n_cluster
        logger.info(f"Loading Siglip model from {SIGLIP_MODEL_PATH}")
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH, use_fast=True)

        # Custom normalization for Siglip
        self.processor.image_processor.image_mean = [0.485, 0.456, 0.406]
        self.processor.image_processor.image_std = [0.229, 0.224, 0.225]
        # Initialize UMAP for dimensionality reduction
        self.reducer = PCA(n_components=3, random_state=0, svd_solver="randomized")
        # Initialize KMeans for clustering
        self.cluster_model = KMeans(
            n_clusters=self.n_cluster, random_state=0, n_init="auto", max_iter=300
        )

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        if len(crops) == 0:
            return np.array([])
        # Convert OpenCV images to PIL format for the processor
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )

                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        if len(crops) == 0:
            raise ValueError("No crops provided for fitting the model.")
            return
        # Extract features from the crops
        logger.info("Extracting features from crops for fitting...")
        data = self.extract_features(crops)
        logger.info("Fitting dimensionality reduction model...")
        projections = self.reducer.fit_transform(data)
        logger.info("Fitting clustering model...")
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])
        logger.info("Extracting features from crops for prediction...")
        data = self.extract_features(crops)
        logger.info("Transforming features with dimensionality reduction...")
        projections = self.reducer.transform(data)
        logger.info("Predicting cluster labels...")
        return self.cluster_model.predict(projections)
