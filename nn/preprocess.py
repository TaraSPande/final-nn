# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Each sequence must correspond to exactly one label
    if len(seqs) != len(labels):
        raise ValueError("seqs and labels must have the same length")

    # Convert to NumPy arrays for easy boolean indexing
    seqs = np.array(seqs)
    # Force boolean dtype so np.where(labels) behaves as expected
    labels = np.array(labels, dtype=bool)

    # Indices of positive examples (labels == True)
    pos_idx = np.where(labels)[0]
    # Indices of negative examples (labels == False)
    neg_idx = np.where(~labels)[0]

    # Count examples per class
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # If a class is empty, cannot sample from it (even with replacement)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must have at least one example to sample from.")

    # Upsample the minority class to match the majority class size
    # This yields a balanced dataset of size 2*target
    target = max(n_pos, n_neg)

    # Use NumPy RNG for sampling and shuffling
    rng = np.random.default_rng()

    # Sample indices for each class
    # If a class is the minority, sample WITH replacement to reach `target`
    # If a class is already at `target`, sample WITHOUT replacement (just a shuffle/subset)
    if n_pos < target:
        sampled_pos_idx = rng.choice(pos_idx, size=target, replace=True)
    else:
        sampled_pos_idx = rng.choice(pos_idx, size=target, replace=False)

    if n_neg < target:
        sampled_neg_idx = rng.choice(neg_idx, size=target, replace=True)
    else:
        sampled_neg_idx = rng.choice(neg_idx, size=target, replace=False)

    # Concatenate positive and negative sampled indices into one index array
    combined_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])

    # Shuffle so that positives and negatives are interleaved (prevents ordered blocks)
    rng.shuffle(combined_idx)

    # Apply the sampled+shuffled indices to sequences and labels
    sampled_seqs = seqs[combined_idx].tolist()
    sampled_labels = labels[combined_idx].tolist()

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # If passes an empty list, return an empty float array
    # This avoids max() failing and is a sensible "no data" encoding
    if not seq_arr:
        return np.empty((0, 0), dtype=np.float32)

    # Normalize sequences:
    # uppercase (so 'a' and 'A' are treated the same)
    # convert None to empty string for robustness
    seqs = [s.upper() if s is not None else "" for s in seq_arr]

    # Pad shorter sequences implicitly with zeros by allocating to max length
    max_len = max(len(s) for s in seqs)

    # n = number of sequences
    n = len(seqs)

    # 3D encoding tensor:
    # enc[i, j, :] is the one-hot vector for base j in sequence i
    # i: sequence index (0..n-1)
    # j: position within sequence (0..max_len-1)
    # 4 channels correspond to A,T,C,G
    enc = np.zeros((n, max_len, 4), dtype=np.float32)

    # Map DNA base characters to channel indices
    # Any character not in this dict (e.g., 'N') will stay all-zeros
    char_to_idx = {
        "A": 0,
        "T": 1,
        "C": 2,
        "G": 3
    }

    # Fill the encoding tensor
    for i, s in enumerate(seqs):
        # Iterate over characters in the sequence
        for j, ch in enumerate(s):
            # Check should never trigger because max_len is the maximum length
            if j >= max_len:
                break

            # Translate base to channel index
            idx = char_to_idx.get(ch)
            if idx is not None:
                # Set the appropriate channel to 1.0 (one-hot)
                enc[i, j, idx] = 1.0
            # else: unknown characters remain zeros (i.e., [0,0,0,0])

    # Flatten the (max_len, 4) grid into a single feature vector per sequence
    # Resulting shape: (n, max_len*4)
    flattened = enc.reshape(n, max_len * 4)
    return flattened