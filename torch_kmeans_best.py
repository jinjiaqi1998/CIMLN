import torch
import numpy as np
import random


class TorchKMeans:
    """
    PyTorch implementation of KMeans clustering algorithm.

    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form.

    max_iter : int, default=300
        Maximum number of iterations for a single run.

    tol : float, default=1e-4
        Tolerance to declare convergence.

    random_state : int, default=None
        Random seed for reproducibility.

    init : {'random', 'k-means++'}, default='k-means++'
        Method for initialization.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    Attributes:
    -----------
    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : torch.Tensor of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, init='k-means++', n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X):
        """Initialize the centroids."""
        n_samples = X.size(0)
        device = X.device

        if self.init == 'random':
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            # Randomly choose k data points as initial centroids
            indices = torch.randperm(n_samples, device=device)[:self.n_clusters]
            centroids = X[indices]

        elif self.init == 'k-means++':
            # Implement k-means++ initialization
            if self.random_state is not None:
                torch.manual_seed(self.random_state)

            # Choose first centroid randomly
            indices = torch.randperm(n_samples, device=device)
            centroids = X[indices[0]].unsqueeze(0)

            # Choose the rest of the centroids
            for _ in range(1, self.n_clusters):
                # Compute squared distances from points to the centroids
                distances = torch.cdist(X, centroids, p=2.0)
                # Get the minimum distance for each point
                min_distances, _ = torch.min(distances, dim=1)
                # Square the distances and normalize to create a probability distribution
                weights = min_distances ** 2
                weights = weights / weights.sum()

                # Choose the next centroid based on the probability distribution
                next_centroid_idx = torch.multinomial(weights, 1)
                next_centroid = X[next_centroid_idx].squeeze(0)

                # Add the new centroid
                centroids = torch.cat([centroids, next_centroid.unsqueeze(0)], dim=0)

        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

        return centroids

    def _compute_inertia(self, X, labels, centroids):
        """Compute the inertia (sum of squared distances to closest centroid)."""
        distances = torch.cdist(X, centroids, p=2.0)
        min_distances = distances[torch.arange(X.size(0), device=X.device), labels]
        return torch.sum(min_distances ** 2).item()

    def _single_kmeans_run(self, X, run_id=0):
        """Perform a single k-means run."""
        device = X.device
        n_samples = X.size(0)

        # Set random seed for this run
        if self.random_state is not None:
            current_seed = self.random_state + run_id
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)

        # Initialize centroids
        centroids = self._init_centroids(X)
        prev_centroids = torch.zeros_like(centroids)

        # For numerical stability
        eps = 1e-8

        # Main loop
        for iteration in range(self.max_iter):
            # Assign samples to closest centroids (E-step)
            distances = torch.cdist(X, centroids, p=2.0)
            labels = torch.argmin(distances, dim=1)

            # Update centroids (M-step)
            prev_centroids = centroids.clone()
            for k in range(self.n_clusters):
                # Select samples that belong to cluster k
                mask = (labels == k)
                cluster_size = mask.sum().item()

                if cluster_size > 0:  # Avoid empty clusters
                    centroids[k] = X[mask].mean(dim=0)
                else:
                    # If a cluster is empty, reinitialize it using a strategy similar to sklearn
                    # Find the point that is furthest from its assigned centroid
                    current_distances = distances[torch.arange(n_samples, device=device), labels]
                    furthest_point = torch.argmax(current_distances)
                    centroids[k] = X[furthest_point]
                    # Update distances and labels for the next iteration
                    distances = torch.cdist(X, centroids, p=2.0)
                    labels = torch.argmin(distances, dim=1)

            # Check for convergence using relative tolerance like sklearn
            centroid_shift = torch.norm(centroids - prev_centroids, dim=1).sum()
            centroid_norm = torch.norm(prev_centroids, dim=1).sum() + eps
            if centroid_shift / centroid_norm < self.tol:
                break

        # Calculate inertia
        inertia = self._compute_inertia(X, labels, centroids)

        return centroids, labels, inertia, iteration + 1

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters:
        -----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training instances to cluster.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        device = X.device

        # Run k-means multiple times and keep the best result
        best_inertia = float('inf')
        best_labels = None
        best_centroids = None
        best_n_iter = 0

        for i in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_kmeans_run(X, run_id=i)

            if inertia < best_inertia:
                best_centroids = centroids
                best_labels = labels
                best_inertia = inertia
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters:
        -----------
        X : torch.Tensor of shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : torch.Tensor of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        distances = torch.cdist(X, self.cluster_centers_, p=2.0)
        return torch.argmin(distances, dim=1)

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters:
        -----------
        X : torch.Tensor of shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : torch.Tensor of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_


def kmeans_torch(X, n_clusters, max_iter=300, tol=1e-4, random_state=None, init='k-means++', n_init=10):
    """
    Convenience function for TorchKMeans with sklearn-like interface.

    Parameters:
    -----------
    X : torch.Tensor of shape (n_samples, n_features)
        Training instances to cluster.

    n_clusters : int
        The number of clusters to form.

    max_iter : int, default=300
        Maximum number of iterations for a single run.

    tol : float, default=1e-4
        Tolerance to declare convergence.

    random_state : int, default=None
        Random seed for reproducibility.

    init : {'random', 'k-means++'}, default='k-means++'
        Method for initialization.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds.

    Returns:
    --------
    centroids : torch.Tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : torch.Tensor of shape (n_samples,)
        Labels of each point.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    kmeans = TorchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        init=init,
        n_init=n_init
    )
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_

