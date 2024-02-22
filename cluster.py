import numpy as np

class cluster:

    def __init__(self):
        self.k = 5
        self.max_iterations = 100
        
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # Initialize centroids randomly
        centroids_indices = np.random.choice(X.shape[0], self.k, replace=False) #X.shape = [rowCount, columnCount]
        centroids = X[centroids_indices]
        prev_centroids = None
        
        # Iterate until convergence or maximum iterations
        for _ in range(self.max_iterations):
            # Assign each instance to the nearest centroid
            ## Add a new axis 
            P = X[:, np.newaxis]
            ## Substract centroids from new added axis data
            T = P - centroids
            ## Normalize
            distances_to_centroids = np.linalg.norm(T, axis=2)
            clusters = np.argmin(distances_to_centroids, axis=1)
            
            # Update centroids
            new_centroids = []
            for i in range(self.k):
                instances_in_cluster = X[clusters == i]
                centroid_i = np.mean(instances_in_cluster, axis=0)
                centroid_i_int = np.round(centroid_i).astype(int)
                new_centroids.append(centroid_i_int)
            new_centroids = np.array(new_centroids)
            
            # Check for convergence
            if prev_centroids is not None and np.array_equal(prev_centroids, new_centroids):
                break
            
            prev_centroids = centroids
            centroids = new_centroids
        
        # Convert results to lists and return
        clusters_list = clusters.tolist()
        centroids_list = centroids.tolist()
        return clusters_list, centroids_list