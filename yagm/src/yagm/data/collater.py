from torch.utils.data import default_collate


def padding_collater(
    samples, max_len, keys=["tokens", "padding_mask"], pad_values=[0, 0], padding=True
):
    """This assume that each Tensor in `keys` has same length at T_dim=0
    """
    first_key = keys[0]
    bs = len(samples)
    sample_lens = [e[first_key].size(0) for e in samples]
    if len(set(sample_lens)) == 1:
        # all has same len, use default collater
        return default_collate(samples)

    padding_samples = [{k: sample.pop(k) for k in keys} for sample in samples]

    batch = default_collate(samples)

    # padding or truncating
    if padding:
        target_len = min(max(sample_lens), max_len)
    else:
        target_len = min(min(sample_lens), max_len)

    for k, pad_v in zip(keys, pad_values):
        batch[k] = padding_samples[0][k].new_full(
            (bs, target_len, *padding_samples[0][k].shape[1:]), pad_v
        )

    for i, (sample, sample_len) in enumerate(zip(padding_samples, sample_lens)):
        diff = target_len - sample_len
        if diff == 0:
            for k in keys:
                batch[k][i] = sample[k]
        elif diff > 0:
            assert padding
            for k in keys:
                batch[k][i, :sample_len] = sample[k]
        else:
            # truncate
            for k in keys:
                batch[k][i] = sample[k][:max_len]

    return batch
