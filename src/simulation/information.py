"""IID binary Experiment 2 models with exact enumerated information (nats)."""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from src.utils.reproducibility import validate_seed


@dataclass(frozen=True)
class InformationModel:
    """bsc: X uniform, Y=X xor Ey; chain/fork: Z uniform, X=Z xor Ex.

    chain uses Y=X xor Ey; fork uses Y=Z xor Ey. All noises and root
    variables are independent, and all rows are IID. Fork has X independent
    of Y GIVEN Z, usually not marginal independence. ex is unused for bsc
    and must explicitly be zero. Endpoint probabilities are supported.
    """
    family: str
    epsilon_x: float
    epsilon_y: float

    def __post_init__(self):
        if self.family not in ('bsc', 'chain', 'fork'):
            raise ValueError('Unknown information model family.')
        for p in (self.epsilon_x, self.epsilon_y):
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Noise probabilities must be finite numbers in [0,1].')
        if self.family == 'bsc' and self.epsilon_x != 0:
            raise ValueError('bsc requires epsilon_x=0 (unused).')

    @property
    def variable_ids(self):
        return ('X', 'Y') if self.family == 'bsc' else ('X', 'Y', 'Z')

    def pmf(self):
        """Enumerate independent root/noise states, including zero-probability support."""
        result = {state: 0. for state in product((0, 1), repeat=len(self.variable_ids))}
        for root, ex, ey in product((0, 1), repeat=3):
            p = .5 * (self.epsilon_x if ex else 1-self.epsilon_x)
            p *= self.epsilon_y if ey else 1-self.epsilon_y
            x = root if self.family == 'bsc' else root ^ ex
            y = (root if self.family == 'fork' else x) ^ ey
            state = (x, y) if self.family == 'bsc' else (x, y, root)
            result[state] += p
        if not math.isclose(math.fsum(result.values()), 1., abs_tol=1e-14):
            raise RuntimeError('Enumerated probabilities are not normalized.')
        return result

    def entropy(self, variables):
        if not variables:
            return 0.
        if len(set(variables)) != len(variables) or not set(variables) <= set(self.variable_ids):
            raise ValueError('Entropy variables must be known and distinct.')
        masses = {}
        for state, p in self.pmf().items():
            key = tuple(state[self.variable_ids.index(v)] for v in variables)
            masses[key] = masses.get(key, 0.) + p
        return -math.fsum(p*math.log(p) for p in masses.values() if p)

    def truths(self):
        """Exact component truths in the same directional convention as the neural API."""
        if self.family == 'bsc':
            first = self.entropy(('X',))
            second = self.entropy(('X', 'Y')) - self.entropy(('Y',))
        else:
            first = self.entropy(('X', 'Z')) - self.entropy(('Z',))
            second = self.entropy(('X', 'Y', 'Z')) - self.entropy(('Y', 'Z'))
        # Algebraic nulls are EXACT zero, not a tolerance-based empirical decision.
        null = self.family == 'fork' or self.epsilon_y == .5 or (
            self.family != 'bsc' and self.epsilon_x in (0, 1))
        if null:
            second = first
        return {'first_entropy_nats': first, 'second_entropy_nats': second,
                'information_nats': 0. if null else first-second, 'is_null': null}

    def generate(self, n, *, seed):
        if type(n) is not int or n <= 0:
            raise ValueError('n must be a positive Python int.')
        validate_seed(seed)
        rng = np.random.default_rng(seed)
        root = rng.binomial(1, .5, n)
        ex = rng.binomial(1, self.epsilon_x, n)
        ey = rng.binomial(1, self.epsilon_y, n)
        x = root if self.family == 'bsc' else root ^ ex
        y = (root if self.family == 'fork' else x) ^ ey
        columns = (x, y) if self.family == 'bsc' else (x, y, root)
        return np.column_stack(columns).astype(np.int64)
