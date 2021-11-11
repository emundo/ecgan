"""Implementations of custom distributions."""
import torch


class TruncatedNormal:
    """
    Sample from a normal distribution truncated to lie within an upper and a lower limit a and b.

    Inspired by https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20.

    Args:
        mu: Mean of the parent normal distribution to sample from.
        sigma: Standard deviation of the parent normal distribution to sample from.
        lower_limit: Lower threshold of the truncated distribution.
        upper_limit: Upper threshold of the truncated distribution.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lower_limit: float = -2.0,
        upper_limit: float = 2.0,
    ):
        self.uniform = torch.distributions.uniform.Uniform(low=0, high=1)
        self.normal = torch.distributions.normal.Normal(0, 1, validate_args=False)
        self.alpha = (lower_limit - mu) / sigma
        self.beta = (upper_limit - mu) / sigma
        self.mu = mu
        self.sigma = sigma
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def sample(self, shape):
        """Generate uniform random variable and apply inverse CDF."""
        # Following https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        # p
        uniform = self.uniform.sample(shape)
        # Φ(μ, σ^2; a)
        alpha_normal_cdf = self.normal.cdf(self.alpha)
        # x =Φ^{−1}(μ, σ^2; Φ(μ, σ^2; a) + p · (Φ(μ, σ^2; b) − Φ(μ, σ^2; a))) which means:
        # Φ(μ, σ^2; a) + p · (Φ(μ, σ^2; b) − Φ(μ, σ^2; a))
        inner_inverse = alpha_normal_cdf + (self.normal.cdf(self.beta) - alpha_normal_cdf) * uniform
        epsilon = torch.finfo(inner_inverse.dtype).eps
        # with x =Φ^{−1}(μ, σ^2;inner_inverse) and numerical stability:
        # erf is not erf(x) but erf(x/sqrt(2)) which will accounted for below
        erf = torch.clamp(2 * inner_inverse - 1, -1 + epsilon, 1 - epsilon)
        # given std normal distribution: samples x = mu + sigma * xi = 0+1*xi = xi clamped to be between given limits
        samples = self.mu + self.sigma * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(erf)
        samples = torch.clamp(samples, self.lower_limit, self.upper_limit)

        return samples
