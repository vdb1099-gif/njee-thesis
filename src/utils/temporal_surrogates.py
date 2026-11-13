"""Path-wise conditional surrogates for the specified binary model, never IID rows."""
import numpy as np

from src.utils.reproducibility import validate_seed


def conditional_source_path(source,target,*,model,direction,cutoff,rng):
    """Keep source prefix through cutoff and ALL target values fixed.

    Forward no-coupling null: X future is its known Markov law conditional on X_V.
    Reverse no-feedback null: Y future is its known coupled law conditional on
    complete autonomous X and Y_V. This retains Y's dependence on X's past.
    The latter is NOT independent shifting of Y. Laws are oracle/model-specific,
    not estimated from final test data; tests concern structural model restrictions
    with sufficient own-history, not arbitrary finite-embedding TE nulls.
    """
    if direction not in ('X_to_Y','Y_to_X') or type(cutoff) is not int or not model.delay<=cutoff<len(source)-1:
        raise ValueError('Invalid direction/prefix cutoff.')
    if (not isinstance(source,np.ndarray) or not isinstance(target,np.ndarray)
            or source.ndim!=1 or target.shape!=source.shape
            or not np.isin(source,(0,1)).all() or not np.isin(target,(0,1)).all()):
        raise ValueError('Surrogates require aligned binary series.')
    result=source.copy()
    for t in range(cutoff+1,len(source)):
        if direction=='X_to_Y':
            p=model.noise_x+(1-2*model.noise_x)*result[t-1]
        else:
            p=model.noise_y+(1-2*model.noise_y)*(
                model.coupling*target[t-model.delay]+(1-model.coupling)*result[t-1])
        result[t]=int(rng.random()<p)
    return result


def circular_shift_source(source,*,shift,minimum_shift):
    """Exploratory only: preserves circular dependence, introduces a wrap seam.

    NOT an exact stationary finite-window conditional-null transformation. No
    calibrated p-value is attached. Source/target-history dependence can break.
    """
    if (not isinstance(source,np.ndarray) or source.ndim!=1 or type(shift) is not int
            or type(minimum_shift) is not int or minimum_shift<1
            or not minimum_shift<=shift<=len(source)-minimum_shift):
        raise ValueError('Shift must leave at least minimum_shift positions on both sides.')
    return np.roll(source,shift)


def block_permute_source(source,*,block_length,seed):
    """Exploratory only; exact within-block order, disrupted between-block transitions.

    Requires complete blocks, without silently truncating. Validity would require
    exchangeable blocks/conditioning assumptions not supplied by this Markov DGP.
    """
    validate_seed(seed)
    if (not isinstance(source,np.ndarray) or source.ndim!=1 or type(block_length) is not int
            or block_length<2 or len(source)%block_length or len(source)<2*block_length):
        raise ValueError('Require at least two complete blocks of length >=2.')
    blocks=source.reshape(-1,block_length)
    return blocks[np.random.default_rng(seed).permutation(len(blocks))].reshape(-1).copy()
