## Final Code
import numpy as np

class SelfOrganizingMap:
    def __init__(self, width, height, input_dim, alpha=0.1, n_iterations=100):
        """
        Initialize the SOM grid and parameters.

        Parameters:
        - width, height: dimensions of the 2D SOM grid
        - input_dim: dimensionality of input vectors
        - alpha: initial learning rate
        - n_iterations: number of training iterations
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.alpha0 = alpha
        self.n_iterations = n_iterations
        
        # Initialize random weights for each node in the grid
        self.weights = np.random.rand(width, height, input_dim)

        # Initial neighborhood radius (half the larger grid dimension)
        self.sigma0 = max(width, height) / 2

        # Time constant for exponential decay of learning rate and radius
        self.lambda_const = n_iterations / np.log(self.sigma0)

    def _get_bmu(self, vector):
        """
        Find the Best Matching Unit (BMU) for a given input vector.

        BMU is the node whose weight vector is closest to the input vector (Euclidean distance).
        """
        distances = np.linalg.norm(self.weights - vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.width, self.height))
        return bmu_idx

    def _decay(self, t):
        """
        Compute the decayed learning rate and neighborhood radius at iteration t.
        """
        sigma_t = self.sigma0 * np.exp(-t / self.lambda_const)
        alpha_t = self.alpha0 * np.exp(-t / self.lambda_const)
        return sigma_t, alpha_t

    def _influence(self, distance, sigma_t):
        """
        Compute the influence of a node based on its grid distance from the BMU.
        Influence decays with distance and time.
        """
        return np.exp(-(distance ** 2) / (2 * sigma_t ** 2))

    def train(self, data):
        """
        Train the SOM using the provided input data.

        For each input vector:
        - Find BMU
        - Update BMU and its neighbors
        - Learning rate and neighborhood shrink over time
        """
        for t in range(self.n_iterations):
            sigma_t, alpha_t = self._decay(t)
            np.random.shuffle(data)  # Shuffle input to reduce order bias
            for vector in data:
                bmu_x, bmu_y = self._get_bmu(vector)
                for x in range(self.width):
                    for y in range(self.height):
                        # Compute distance from current node to BMU on the grid
                        dist_to_bmu = np.sqrt((x - bmu_x) ** 2 + (y - bmu_y) ** 2)
                        if dist_to_bmu <= sigma_t:
                            theta = self._influence(dist_to_bmu, sigma_t)
                            # Update weights of nodes within the neighborhood
                            self.weights[x, y] += alpha_t * theta * (vector - self.weights[x, y])

    def map_input(self, data):
        """
        Map each input vector to its corresponding BMU on the trained grid.
        Returns a list of grid coordinates.
        """
        mapped = []
        for vector in data:
            bmu_idx = self._get_bmu(vector)
            mapped.append(bmu_idx)
        return mapped

    def get_weights(self):
        """
        Return the trained weight vectors of the SOM grid.
        """
        return self.weights


        
