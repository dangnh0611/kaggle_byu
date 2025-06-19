from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


def _extend_list(li, offset=0):
    ret = list(range(offset))
    ret.extend([e + offset for e in li])
    return ret


class BaseTTA:

    def transform(self, x: Tensor) -> Tuple[Tensor, Union[Tensor, float]]:
        raise NotImplementedError

    def invert(self, x: Tensor) -> Tuple[Tensor, Union[Tensor, float]]:
        raise NotImplementedError


class PermuteTTA(BaseTTA):
    """
    Simple dimensionals permuting using torch.permute()
    e.g, 6 posible permutations: xyz, xzy, yxz, yzx, zxy, zyx for 3D case

    Args:
        code: permutation ordering, relative to `ori_dims`
        ori_dims: notation of original dimension names
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        self.ndim = len(ori_dims)
        permute_dims = [ori_dims.index(e) for e in code]
        assert sorted(permute_dims) == list(range(self.ndim))
        self.permute_dims = permute_dims
        self.inv_permute_dims = [permute_dims.index(e) for e in list(range(self.ndim))]
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.permute(x, _extend_list(self.permute_dims, offset))
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.permute(x, _extend_list(self.inv_permute_dims, offset))
        return x, self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"


class PermuteByRot90TTA(BaseTTA):
    """
    Use consecutive rot90() to archived shape specified by `code`,
    butkeep handedness (no flip operation).
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        self.ndim = len(ori_dims)
        permute_dims = [ori_dims.index(e) for e in code]
        assert sorted(permute_dims) == list(range(self.ndim))
        rot_params = self.min_swaps_planing(permute_dims)
        self.rot_params = rot_params
        self.weight = weight
        self.ori_dims = ori_dims

    def min_swaps_planing(self, L):
        """
        Finds the minimum number of swaps to sort a list of consecutive numbers 0..N-1.

        Args:
            L: A list of N consecutive numbers 0..N-1 in random order.

        Returns:
            A list of tuples (i, j) representing the swap operations, or None if input is invalid.
            Returns empty list if the input is already sorted.
        """

        n = len(L)
        if n == 0:
            return []

        if not all(
            0 <= x < n for x in L
        ):  # check if all elements are in the range 0..N-1
            return None  # Or raise an exception: raise ValueError("Invalid input list: elements not in range 0..N-1")
        if len(set(L)) != n:
            return None  # check if all elements are distinct

        swaps = []
        visited = [False] * n

        for i in range(n):
            if visited[i] or L[i] == i:  # Already visited or in correct position
                continue

            cycle_size = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = L[j]  # No need to subtract 1 now
                cycle_size += 1

            if cycle_size > 1:  # Need swaps if cycle has more than one element
                j = i
                for _ in range(cycle_size - 1):
                    k = L[j]
                    swaps.append((j, k))
                    j = k
        return swaps

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        for dims in self.rot_params:
            dims = [d + offset for d in dims]
            x = torch.rot90(x, k=1, dims=dims)
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        for dims in self.rot_params[::-1]:
            dims = [d + offset for d in dims]
            x = torch.rot90(x, k=3, dims=dims)
        return x, self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"


class FlipTTA(BaseTTA):
    """
    Flip TTA: 8 posible Flipping: identity, flip along one of [x, y, z, xy, xz, yz, xyz]

    Args:
        code: code represent the flipping dims, relative to `ori_dims`
        ori_dims: notation of original dimension names
    """

    def __init__(self, code: str, weight: float, ori_dims="xyz"):
        self.ndim = len(ori_dims)
        flip_dims = [ori_dims.index(e) for e in code]
        assert len(flip_dims) == len(set(flip_dims))
        self.flip_dims = flip_dims
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def transform(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.flip(x, _extend_list(self.flip_dims, offset))
        return x, self.weight

    def invert(self, x):
        offset = len(x.shape) - self.ndim
        assert offset >= 0
        x = torch.flip(x, _extend_list(self.flip_dims, offset))
        return x, self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"


class PermuteFlipTTA(BaseTTA):
    """
    Permute + Flip TTA:
    - 6 posible permutations (rot90): xyz, xzy, yxz, yzx, zxy, zyx
    - 8 posible Flipping: identity, flip along one of [x, y, z, xy, xz, yz, xyz]
    Total: 6 * 8 = 48 posibles unique transformations

    Args:
        code: code represent the transform in form of `{permute_dims}_{flip_dims}`, e.g `zxy_xy`
        ori_dims: notation of original dimension names
    """

    def __init__(
        self, code: str, weight: float, ori_dims="xyz", permute_keep_handedness=True
    ):
        self.ndim = len(ori_dims)
        splits = code.split("_")
        if len(splits) == 1:
            permute_code = splits[0]
            flip_code = ""
        elif len(splits) == 2:
            permute_code, flip_code = splits
        else:
            raise ValueError

        if permute_keep_handedness:
            self.permute_tta = PermuteByRot90TTA(permute_code, weight, ori_dims)
        else:
            self.permute_tta = PermuteTTA(permute_code, weight, ori_dims)
        self.flip_tta = FlipTTA(flip_code, weight, ori_dims)
        self.code = code
        self.weight = weight
        self.ori_dims = ori_dims

    def transform(self, x):
        x, _ = self.permute_tta.transform(x)
        x, _ = self.flip_tta.transform(x)
        return x, self.weight

    def invert(self, x):
        x, _ = self.flip_tta.invert(x)
        x, _ = self.permute_tta.invert(x)
        return x, self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, weight={self.weight}, ori_dims={self.ori_dims})"


def build_tta(code, weight, ori_dims="xyz"):
    return PermuteFlipTTA(code, weight, ori_dims=ori_dims, permute_keep_handedness=True)
