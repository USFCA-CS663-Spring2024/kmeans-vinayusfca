import numpy as np

class cluster:

    def __init__(self):
        self.k = 5
        self.max_iterations = 100
        
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # Randomly initialize centroids
        # centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        # prev_centroids = None
        
        # for _ in range(self.max_iterations):
        #     # Assign each instance to the nearest centroid
        #     clusters = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            
        #     # Update centroids
        #     new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
            
        #     # Check for convergence
        #     if np.array_equal(prev_centroids, new_centroids):
        #         break
                
        #     prev_centroids = centroids
        #     centroids = new_centroids
        
        # return clusters.tolist(), centroids.tolist()
        # Initialize centroids randomly
        # print("Ikda::", X.shape)
        centroids_indices = np.random.choice(X.shape[0], self.k, replace=False) #X.shape = [rowCount, columnCount]
        # print("Ikda2::", centroids_indices)
        centroids = X[centroids_indices]
        # print("Ikda3::", centroids)
        prev_centroids = None
        
        # Iterate until convergence or maximum iterations
        for _ in range(self.max_iterations):
            # Assign each instance to the nearest centroid
            print("X:",X)
            ## Add a new axis 
            P = X[:, np.newaxis]
            print("PP::",P[0])
            ## Substract centroids from new added axis data
            T = P - centroids
            print("T::",T[0])
            ## Normalize
            distances_to_centroids = np.linalg.norm(T, axis=2)
            print("distances_to_centroids:",distances_to_centroids)
            clusters = np.argmin(distances_to_centroids, axis=1)
            print("clusters::", clusters)
            
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



# if __name__ == "__main__":
#     X = np.array([[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]])
#     k = 2
#     kmeans = cluster(k=k, max_iterations =100)
#     cluster_hypotheses, centroids = kmeans.fit(X)
#     print("Cluster hypotheses:", cluster_hypotheses)
#     print("Centroids:", centroids)