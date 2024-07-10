import numpy as np
from scipy.stats import multivariate_normal

# class GaussianMixtureModelRandom:
#     """This class generates a random Gaussian Mixture Model (GMM) with a specified number of components and in a given space. It provides a method to 
#     calculate the probability density function (PDF) of the GMM at a given point.

#         Args:
#             n_components_range (tuple): A tuple specifying the range of possible values for the number of components in the GMM.
#             space_range (tuple): A tuple specifying the range of values for the space in which the GMM is defined.

#         Attributes:
#             n_components_range (tuple): The range of possible values for the number of components in the GMM.
#             space_range (tuple): The range of values for the space in which the GMM is defined.
#             n_components (int): The randomly selected number of components for the GMM.
#             means (list): A list of randomly generated mean vectors for each component in the GMM.
#             covariances (list): A list of randomly generated covariance matrices for each component in the GMM.
#             weights (array): An array of randomly generated weights representing the mixture proportions of each component in the GMM.

#         Methods:
#             pdf(x): Calculate the probability density function (PDF) of the GMM at a given point.
#     """
#     def __init__(self, n_components_range, space_range):
#         self.n_components_range = n_components_range
#         self.space_range = space_range
#         self.n_components = np.random.randint(*n_components_range)
#         self.means = [np.random.uniform(*space_range, size=2) for _ in range(self.n_components)]
#         self.covariances = [np.eye(2) * np.random.uniform(0.1, 2) for _ in range(self.n_components)]
#         self.weights = np.random.dirichlet(np.ones(self.n_components))

#     def pdf(self, x):
#         # Calculate the PDF of the GMM at x
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

class ModifiedGaussianMixtureModelRandom:
    """
            This class generates a random Gaussian Mixture Model (GMM) with a specified 
            number of components and in a given space, ensuring that the means of the Gaussian 
            components are spaced apart by at least a distance r. It provides a method to calculate 
            the probability density function (PDF) of the GMM at a given point.

        Args:
            n_components_range (tuple): A tuple specifying the range of possible values for the number of components in the GMM.
            space_range (tuple): A tuple specifying the range of values for the space in which the GMM is defined.
            covariance_range (tuple): A tuple specifying the range of values for the covariance matrix. 
            r (float, optional): The exclusion radius ensuring that Gaussian means are at least this distance apart. Defaults to 2.
            max_attempts (int, optional): The maximum number of attempts to find a mean that satisfies the distance constraint. Defaults to 10.
            decrease_factor (float, optional): Factor by which r is decreased if a suitable mean cannot be found within max_attempts. Defaults to 0.9.

        Attributes:
            n_components_range (tuple): The range of possible values for the number of components in the GMM.
            space_range (tuple): The range of values for the space in which the GMM is defined.
            n_components (int): The randomly selected number of components for the GMM.
            means (list): A list of randomly generated mean vectors for each component in the GMM, ensuring that they are spaced apart by at least a distance r.
            covariances (list): A list of randomly generated covariance matrices for each component in the GMM.
            weights (array): An array of randomly generated weights representing the mixture proportions of each component in the GMM.
            r (float): The exclusion radius.
            max_attempts (int): The maximum number of attempts.
            decrease_factor (float): The factor by which r is decreased in case of difficulties in finding a new mean.

        Methods:
            pdf(x): Calculate the probability density function (PDF) of the GMM at a given point.
    """
    def __init__(self, n_components_range, covariance_range, space_range, r=2, max_attempts=10, decrease_factor=0.9, seed=None, min_weight=0.15):
        # Setting the seed for reproducibility
        np.random.seed(seed)
        self.n_components_range = n_components_range
        self.space_range = space_range
        self.r = r
        self.max_attempts = max_attempts
        self.decrease_factor = decrease_factor
        
        self.n_components = np.random.randint(*n_components_range)
        
        # Generate means ensuring they're spaced apart by at least r
        self.means = self.generate_means()
        
        # Other parameters remain similar to original implementation
        self.covariances = [np.eye(2) * np.random.uniform(covariance_range[0], covariance_range[1]) for _ in range(self.n_components)]
        
        # Generate weights ensuring none are below the minimum threshold
        self.weights = self.generate_weights(min_weight)

    def generate_means(self):
        means = []
        for _ in range(self.n_components):
            valid = False
            attempts = 0
            while not valid and attempts < self.max_attempts:
                candidate_mean = np.random.uniform(*self.space_range, size=2)
                if all(np.linalg.norm(candidate_mean - mean) >= self.r for mean in means):
                    valid = True
                    means.append(candidate_mean)
                attempts += 1
            
            # If we exhausted our attempts, decrease r and try again
            if not valid:
                self.r *= self.decrease_factor
        return means

    def generate_weights(self, min_weight):
        weights = np.random.dirichlet(np.ones(self.n_components))
        while any(weight < min_weight for weight in weights):
            weights = np.random.dirichlet(np.ones(self.n_components))
        return weights

    def pdf(self, x):
        # Calculate the PDF of the GMM at x
        pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
        return pdf
    