import numpy as np
import numpy.random as npr
import numpy.typing as npt
import abc
import typing

# the (pesudo) random number generator
rng = npr.Generator(npr.MT19937(19260817))


def natural_number_stream(start: int = 0) -> typing.Iterator[int]:
    r'''
    generate a natural number stream [start, start+1, start+2, ...]
    '''
    while True:
        yield start
        start += 1


def argmax(xs: typing.Iterable) -> int:
    r'''
    index of the maximum element
    '''
    return max(zip(natural_number_stream(), xs), key=lambda kv: kv[1])[0]


def argmin(xs: typing.Iterable) -> int:
    r'''
    index of the minimum element
    '''
    return max(zip(natural_number_stream(), xs), key=lambda kv: -kv[1])[0]


class Sampling:
    r'''
    sampling named distribution

    methods for single sample generation:
    - uniform(a,b):        $\mathrm{Unif}(a,b)$
    - bernoulli(p):        $\mathrm{Bern}(p)$
    - exponential(rate):   $\mathrm{Expo}(\lambda)$, where rate is the $\lambda$ parameter.
    - gamma(n,rate):       $\mathrm{Gamma}(a,\lambda)$, where $a$ is $n$ and $\lambda$ is rate.

    methods for multiple i.i.d. samples generation:
    - uniform_array
    - bernoulli_array
    - exponential_array
    '''

    @staticmethod
    def uniform(a: float = 0, b: float = 1) -> float:
        return a+(b-a)*rng.random()

    @staticmethod
    def uniform_array(size: int, a: float = 0, b: float = 1) -> npt.NDArray[np.float_]:
        return a+(b-a)*rng.random(size)

    @staticmethod
    def bernoulli(prob: float) -> bool:
        return Sampling.uniform() < prob

    @staticmethod
    def bernoulli_array(size: int, prob: float) -> npt.NDArray[np.bool_]:
        return Sampling.uniform_array(size) < prob

    @staticmethod
    def exponential(rate: float = 1) -> float:
        return -np.log(1-Sampling.uniform())/rate

    @staticmethod
    def exponential_array(size: int, rate: float = 1) -> npt.NDArray[np.float_]:
        return -np.log(1-Sampling.uniform_array(size))/rate

    @staticmethod
    def gamma(n: int, rate: float = 1) -> float:
        return Sampling.exponential_array(n, rate).sum()

    @staticmethod
    def beta(a: int, b: int, _rate: float = 1000) -> float:
        x, y = Sampling.gamma(a, _rate), Sampling.gamma(b, _rate)
        return x/(x+y)


class MAB:
    r'''
    the multi-armed bandit
    '''

    def __init__(self, theta: list[float]):
        self.theta = theta[:]
        self.arms = len(theta)

    def pull(self, i: int) -> int:
        return int(Sampling.bernoulli(self.theta[i]))

    def oracle_value(self, n: int) -> float:
        return n*max(self.theta)


class Strategy(abc.ABC):
    r'''
    the abstract base class for bandit algorithms
    '''

    def __init__(self, mab: MAB, n: int):
        self.mab = mab
        self.n = n

    @abc.abstractmethod
    def run(self) -> int:
        '''
        perform one simulation: n pulls
        '''
        ...

    @property
    @abc.abstractmethod
    def profile(self) -> str:
        '''
        return the strategy/algorithm name and value of parameters
        '''
        ...


########################################################################################


class EpsilonGreedy(Strategy):
    def __init__(self, mab: MAB, n: int, eps: float):
        super().__init__(mab, n)
        self._profile = f'EpsilonGreedy(epsilon={eps})'
        self.eps = eps
        self.count = [int(0) for _ in range(mab.arms)]
        self.theta_hat = [float(0) for _ in range(mab.arms)]

    def run(self) -> int:
        earn = 0
        for _ in range(self.n):
            arm = argmax(self.count)
            if Sampling.bernoulli(self.eps):
                arm = rng.integers(self.mab.arms)
            reward = self.mab.pull(arm)
            earn += reward

            self.count[arm] += 1
            self.theta_hat[arm] += (reward-self.theta_hat[arm])/self.count[arm]

        return earn

    @property
    def profile(self) -> str:
        return self._profile


class UpperConfidenceBound(Strategy):
    def __init__(self, mab: MAB, n: int, c: float):
        super().__init__(mab, n)
        self._profile = f'UpperConfidenceBound(c={c})'
        self.c = c
        self.count = [int(0) for _ in range(mab.arms)]
        self.theta_hat = [float(0) for _ in range(mab.arms)]

    def run(self) -> int:
        from math import log, sqrt
        earn = 0
        for t in range(self.mab.arms):
            reward = self.mab.pull(t)
            earn += reward

            self.count[t] = 1
            self.theta_hat[t] = reward

        for t in range(self.mab.arms, self.n):
            arm = argmax(self.theta_hat[i] + self.c*sqrt(2*log(t+1)/self.count[i])
                         for i in range(self.mab.arms))
            reward = self.mab.pull(arm)
            earn += reward

            self.count[arm] += 1
            self.theta_hat[arm] += (reward-self.theta_hat[arm])/self.count[arm]

        return earn

    @property
    def profile(self) -> str:
        return self._profile


class ThompsonSampling(Strategy):
    def __init__(self, mab: MAB, n: int, prior: list[tuple[int, int]]):
        super().__init__(mab, n)
        self.beta_parameters = [list(i) for i in prior]
        self._profile = f'ThompsonSampling({prior})'

    def run(self) -> int:
        earn = 0
        for _ in range(self.n):
            theta_hat = [Sampling.beta(a, b)
                         for (a, b) in self.beta_parameters]
            arm = argmax(theta_hat)
            reward = self.mab.pull(arm)
            earn += reward

            self.beta_parameters[arm][0] += reward
            self.beta_parameters[arm][0] += 1-reward

        return earn

    @property
    def profile(self) -> str:
        return self._profile

########################################################################################


class EpsilonDecreaseGreedy(Strategy):
    def __init__(self, mab: MAB, n: int, eps: float, shrink_factor: float):
        super().__init__(mab, n)
        self._profile = f'EpsilonDecreaseGreedy(epsilon={eps}, shrink_factor={shrink_factor})'
        self.eps = eps
        self.shrink_factor = shrink_factor
        self.count = [int(0) for _ in range(mab.arms)]
        self.theta_hat = [float(0) for _ in range(mab.arms)]

    def run(self) -> int:
        earn = 0
        for _ in range(self.n):
            arm = argmax(self.count)
            if Sampling.bernoulli(self.eps):
                arm = rng.integers(self.mab.arms)
            reward = self.mab.pull(arm)
            earn += reward

            self.count[arm] += 1
            self.theta_hat[arm] += (reward-self.theta_hat[arm])/self.count[arm]

            self.eps *= self.shrink_factor

        return earn

    @property
    def profile(self) -> str:
        return self._profile


RUNS = 200
N = 6000
mab = MAB([0.8, 0.6, 0.5])


def once() -> dict[str, int]:
    eps_gre: list[Strategy] = [
        EpsilonGreedy(mab, N, 0.2),
        EpsilonGreedy(mab, N, 0.4),
        EpsilonGreedy(mab, N, 0.6),
        EpsilonGreedy(mab, N, 0.8),
    ]
    eps_dec_gre: list[Strategy] = [
        EpsilonDecreaseGreedy(mab, N, 1, 0.95),
        EpsilonDecreaseGreedy(mab, N, 0.5, 0.95),
        EpsilonDecreaseGreedy(mab, N, 0.2, 0.99),
    ]
    ucb: list[Strategy] = [
        UpperConfidenceBound(mab, N, 1),
        UpperConfidenceBound(mab, N, 2),
        UpperConfidenceBound(mab, N, 6),
        UpperConfidenceBound(mab, N, 9),
    ]
    ts: list[Strategy] = [
        ThompsonSampling(mab, N, [(1, 1), (1, 1), (1, 1)]),
        ThompsonSampling(mab, N, [(601, 401), (401, 601), (2, 3)]),
        ThompsonSampling(mab, N, [(8, 2), (6, 4), (5, 5)]),
        ThompsonSampling(mab, N, [(80, 20), (6, 4), (5, 5)]),
    ]
    all_in_one = eps_gre + eps_dec_gre + ucb + ts

    return {
        s.profile: s.run()
        for s in all_in_one
    }


rec = None
for _ in range(RUNS):
    out = once()
    if rec is None:
        rec = {k: [v] for (k, v) in out.items()}
    else:
        for (k, v) in out.items():
            rec[k].append(v)
