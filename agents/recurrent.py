import torch
from torch import nn

def unwrap_packed_sequences_recursive(packed):
    """Unwrap `PackedSequence` class of packed sequences recursively.

    This function extract `torch.Tensor` that
    `torch.nn.utils.rnn.PackedSequence` holds internally. Sequences in the
    internal tensor is ordered with time axis first.

    Unlike `torch.nn.pad_packed_sequence`, this function just returns the
    underlying tensor as it is without padding.

    To wrap the data by `PackedSequence` again, use
    `wrap_packed_sequences_recursive`.

    Args:
        packed (object): Packed sequences.

    Returns:
        object: Unwrapped packed sequences. If `packed` is a `PackedSequence`,
            then the returned value is `PackedSequence.data`, the underlying
            tensor. If `Packed` is a tuple of `PackedSequence`, then the
            returned value is a tuple of the underlying tensors.
    """
    if isinstance(packed, torch.nn.utils.rnn.PackedSequence):
        return packed.data
    if isinstance(packed, tuple):
        return tuple(unwrap_packed_sequences_recursive(x) for x in packed)
    return packed

def pack_sequences_recursive(sequences):
    """Pack sequences into PackedSequence recursively.

    This function works similarly to `torch.nn.utils.rnn.pack_sequence` except
    that it works recursively for tuples.

    When each given sequence is an N-tuple of `torch.Tensor`s, the function
    returns an N-tuple of `torch.nn.utils.rnn.PackedSequence`, packing i-th
    tensors separately for i=1,...,N.

    Args:
        sequences (object): Batch of sequences to pack.

    Returns:
        object: Packed sequences. If `sequences` is a list of tensors, then the
            returned value is a `PackedSequence`. If `sequences` is a list of
            tuples of tensors, then the returned value is a tuple of
            `PackedSequence`.
    """
    assert sequences
    first_seq = sequences[0]
    if isinstance(first_seq, torch.Tensor):
        return nn.utils.rnn.pack_sequence(sequences)
    if isinstance(first_seq, tuple):
        return tuple(
            pack_sequences_recursive([seq[i] for seq in sequences])
            for i in range(len(first_seq))
        )
    return sequences

def pack_and_forward(rnn, sequences, recurrent_state):
    """Pack sequences, multi-step forward, and then unwrap `PackedSequence`.

    Args:
        rnn (torch.nn.Module): Recurrent module.
        sequences (object): Sequences of input data.
        recurrent_state (object): Batched recurrent state.

    Returns:
        object: Sequence of output data, packed with time axis first.
        object: New batched recurrent state.
    """
    pack = pack_sequences_recursive(sequences)
    y, recurrent_state = rnn(pack, recurrent_state)
    return unwrap_packed_sequences_recursive(y), recurrent_state