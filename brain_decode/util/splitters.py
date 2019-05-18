import numpy as np
from util.util import apply_to_X_y, SignalAndTarget


def get_balanced_batches(n_trials, rng, shuffle, n_batches=None, batch_size=None):
    """Create indices for batches balanced in size
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).
    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional
    Returns
    -------
    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


def concatenate_sets(sets):
    """
    Concatenate all sets together.

    Parameters
    ----------
    sets: list of :class:`.SignalAndTarget`
    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    concatenated_set = sets[0]
    for s in sets[1:]:
        concatenated_set = concatenate_two_sets(concatenated_set, s)
    return concatenated_set


def concatenate_two_sets(set_a, set_b):
    """
    Concatenate two sets together.

    Parameters
    ----------
    set_a, set_b: :class:`.SignalAndTarget`
    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    new_X = concatenate_np_array_or_add_lists(set_a.X, set_b.X)
    new_y = concatenate_np_array_or_add_lists(set_a.y, set_b.y)
    return SignalAndTarget(new_X, new_y)


def concatenate_np_array_or_add_lists(a, b):
    if hasattr(a, "ndim") and hasattr(b, "ndim"):
        new = np.concatenate((a, b), axis=0)
    else:
        if hasattr(a, "ndim"):
            a = a.tolist()
        if hasattr(b, "ndim"):
            b = b.tolist()
        new = a + b
    return new


def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set
    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (
        n_first_set is None
    ), "Pass either first_set_fraction or n_first_set"
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y(lambda a: a[n_first_set:], dataset)
    return first_set, second_set


def select_examples(dataset, indices):
    """
    Select examples from dataset.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    indices: list of int, 1d-array of int
        Indices to select
    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    # probably not necessary
    indices = np.array(indices)
    if hasattr(dataset.X, "ndim"):
        # numpy array
        new_X = np.array(dataset.X)[indices]
    else:
        # list
        new_X = [dataset.X[i] for i in indices]
    new_y = np.asarray(dataset.y)[indices]
    return SignalAndTarget(new_X, new_y)


def split_into_train_valid_test(dataset, n_folds, i_test_fold, rng=None):
    """
    Split datasets into folds, select one valid fold, one test fold and
    merge rest as train fold.
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    n_folds: int
        Number of folds to split dataset into.
    i_test_fold: int
        Index of the test fold (0-based). Validation fold will be
        immediately preceding fold.
    rng: `numpy.random.RandomState`, optional
        Random Generator for shuffling, None means no shuffling
    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    n_trials = len(dataset.X)
    if n_trials < n_folds:
        raise ValueError("Less Trials: {:d} than folds: {:d}".format(n_trials, n_folds))
    shuffle = rng is not None
    folds = get_balanced_batches(n_trials, rng, shuffle, n_batches=n_folds)
    test_inds = folds[i_test_fold]
    valid_inds = folds[i_test_fold - 1]
    all_inds = list(range(n_trials))
    train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
    assert np.intersect1d(train_inds, valid_inds).size == 0
    assert np.intersect1d(train_inds, test_inds).size == 0
    assert np.intersect1d(valid_inds, test_inds).size == 0
    assert np.array_equal(
        np.sort(np.union1d(train_inds, np.union1d(valid_inds, test_inds))), all_inds
    )

    train_set = select_examples(dataset, train_inds)
    valid_set = select_examples(dataset, valid_inds)
    test_set = select_examples(dataset, test_inds)

    return train_set, valid_set, test_set


def split_into_train_test(dataset, n_folds, i_test_fold, rng=None):
    """
     Split datasets into folds, select one test fold and merge rest as train fold.
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    n_folds: int
        Number of folds to split dataset into.
    i_test_fold: int
        Index of the test fold (0-based)
    rng: `numpy.random.RandomState`, optional
        Random Generator for shuffling, None means no shuffling
    Returns
    -------
    reduced_set: :class:`.SignalAndTarget`
        Dataset with only examples selected.
    """
    n_trials = len(dataset.X)
    if n_trials < n_folds:
        raise ValueError("Less Trials: {:d} than folds: {:d}".format(n_trials, n_folds))
    shuffle = rng is not None
    folds = get_balanced_batches(n_trials, rng, shuffle, n_batches=n_folds)
    test_inds = folds[i_test_fold]
    all_inds = list(range(n_trials))
    train_inds = np.setdiff1d(all_inds, test_inds)
    assert np.intersect1d(train_inds, test_inds).size == 0
    assert np.array_equal(np.sort(np.union1d(train_inds, test_inds)), all_inds)

    train_set = select_examples(dataset, train_inds)
    test_set = select_examples(dataset, test_inds)
    return train_set, test_set
