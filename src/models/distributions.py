import torch
import torch.nn.functional as F
import torch.distributions
import numpy as np


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class SafeTruncatedNormal(torch.distributions.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)

class MyRelaxedOneHotCategorical(torch.distributions.RelaxedOneHotCategorical):
    def __init__(self, temp, logits, eps=1e-16, validate_args=False):
        super(MyRelaxedOneHotCategorical, self).__init__(temp, logits=logits, validate_args=validate_args)
        """
        re-write the rsample() api.
        """
        self.dev = logits.device
        self.eps = eps

    def log_prob(self, value):
        K = self.logits.shape[-1]

        log_scale = torch.full_like(self.temperature, float(K)).lgamma() - self.temperature.log().mul(-(K - 1))
        score = self.logits - value.log().mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)

        log_prob = score + log_scale
        if torch.isinf(log_prob).any():
            pdb.set_trace()

        return log_prob

    def rsample(self, eps=1e-20):
        uniforms = torch.rand(self.logits.shape, dtype=self.logits.dtype, device=self.dev)
        uniforms = torch.clamp(uniforms, self.eps, 1.0 - self.eps)
        gumbels = -torch.log(-torch.log(uniforms))
        scores = (self.logits + gumbels) / self.temperature
        samples = (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()
        if torch.isnan(samples).any():
            pdb.set_trace()
        # if (1. - self.support.check(samples).float()).any():
        #   pdb.set_trace()
        return samples