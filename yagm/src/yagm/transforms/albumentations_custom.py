from __future__ import annotations

import math
import random
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, TypedDict, cast
from warnings import warn

import albumentations as A
import cv2
import numpy as np
from albumentations import random_utils
from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.functional import (
    rotation2d_matrix_to_euler_angles,
)
from albumentations.augmentations.utils import angle_2pi_range, handle_empty_array
from albumentations.core.bbox_utils import union_of_bboxes
from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    NonNegativeIntRangeType,
    OnePlusIntRangeType,
    ProbabilityType,
    ZeroOneRangeType,
    check_0plus,
    check_01,
    check_1plus,
    nondecreasing,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.types import (
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    ColorType,
    ScaleIntType,
    Targets,
)
from pydantic import AfterValidator, Field, field_validator, model_validator
from typing_extensions import Annotated, Self
import albumentations as A

# @TODO - pull request this
# THERE ARE BUG IN ALBUMENTATIONS IMPLEMENTATION WHEN NUMBER OF KEYPOINTS EQUAL TO 1
@handle_empty_array
@angle_2pi_range
def perspective_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> np.ndarray:
    keypoints = keypoints.copy().astype(np.float32)
    x, y, angle, scale = (
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        keypoints[:, 3],
    )

    # Reshape keypoints for perspective transform
    keypoint_vector = np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2)

    # Apply perspective transform
    # ---original implementation
    # transformed_points = cv2.perspectiveTransform(keypoint_vector, matrix).squeeze()
    # +++ change to
    transformed_points = cv2.perspectiveTransform(keypoint_vector, matrix).squeeze(1)
    x, y = transformed_points[:, 0], transformed_points[:, 1]

    # Update angles
    angle += rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=True)

    # Calculate scale factors
    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        width, height = image_shape[:2]
        scale_x = width / max_width
        scale_y = height / max_height
        x *= scale_x
        y *= scale_y
        scale *= max(scale_x, scale_y)

    # Create the output array
    transformed_keypoints = np.column_stack([x, y, angle, scale])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        transformed_keypoints = np.column_stack(
            [
                transformed_keypoints,
                keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:],
            ],
        )

    return transformed_keypoints


A.augmentations.geometric.functional.perspective_keypoints = perspective_keypoints


class _CustomBaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(
        self,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p, always_apply)

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        crop = fcrops.crop(img, *crop_coords)
        return crop

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        keypoints = fcrops.crop_keypoints_by_coords(keypoints, crop_coords)
        return keypoints


class CustomRandomSizedBBoxSafeCrop(_CustomBaseRandomSizedCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        size (int, int): expected output size of the crop, for each edge. If size is an int instead of sequence
            like (height, width), a square output size (size, size) is made. If provided a sequence of length 1,
            it will be interpreted as (size[0], size[0]).
        scale ((float, float)): Specifies the lower and upper bounds for the random area of the crop, before resizing.
            The scale is defined with respect to the area of the original image.
        ratio ((float, float)): lower and upper bounds for the random aspect ratio of the crop, before resizing.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        crop_size: Optional[tuple[int, int]] = None
        scale: Annotated[tuple[float, float], AfterValidator(check_01)] = (0.08, 1.0)
        ratio: Annotated[tuple[float, float], AfterValidator(check_0plus)] = (
            0.75,
            1.3333333333333333,
        )
        get_bbox_func: Optional[Callable] = None
        retry: int = 30
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def process(self) -> Self:
            return self

    def __init__(
        self,
        crop_size: Optional[tuple[int, int]] = None,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        get_bbox_func: Optional[Callable] = None,
        retry: int = 30,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        if get_bbox_func is None:
            self.get_bbox_func = self._default_get_bbox
        else:
            self.get_bbox_func = get_bbox_func
        self.retry = retry

    def _default_get_bbox(self, params, data):
        """
        Return union of all bboxes in form (x_min, y_min, x_max, y_max).
        Return None if no bbox is provided.
        """
        if (
            "bboxes" not in data or len(data["bboxes"]) == 0
        ):  # less likely, this class is for use with bboxes.
            return None
        else:
            bbox_union = union_of_bboxes(bboxes=data["bboxes"], erosion_rate=0.0)

        x_min, y_min, x_max, y_max = bbox_union
        x_min = np.clip(x_min, 0, 1)
        y_min = np.clip(y_min, 0, 1)
        x_max = np.clip(x_max, x_min, 1)
        y_max = np.clip(y_max, y_min, 1)
        image_height, image_width = params["shape"][:2]
        return [
            int(x_min * image_width),
            int(y_min * image_height),
            int(x_max * image_width),
            int(y_max * image_height),
        ]

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_height, image_width = params["shape"][:2]
        area = image_height * image_width

        keep_bbox = self.get_bbox_func(params, data)

        for _ in range(self.retry):
            if self.crop_size is not None:
                height, width = self.crop_size
            else:
                target_area = random.uniform(*self.scale) * area
                log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))

                width = int(round(math.sqrt(target_area * aspect_ratio)))
                height = int(round(math.sqrt(target_area / aspect_ratio)))

            can_continue = 0 < width <= image_width and 0 < height <= image_height

            if can_continue:
                if keep_bbox is not None:
                    keep_x_min, keep_y_min, keep_x_max, keep_y_max = [
                        int(e) for e in keep_bbox
                    ]
                    start_y_min = max(0, keep_y_max - height)
                    start_y_max = min(keep_y_min, image_height - height)
                    start_x_min = max(0, keep_x_max - width)
                    start_x_max = min(keep_x_min, image_width - width)
                else:
                    start_y_min = 0
                    start_y_max = image_height - height
                    start_x_min = 0
                    start_x_max = image_width - width
                can_continue = (
                    can_continue
                    and start_x_min <= start_x_max
                    and start_y_min <= start_y_max
                )

            if can_continue:
                i = random.randint(start_y_min, start_y_max)
                j = random.randint(start_x_min, start_x_max)

                # print('OK:', image_width, image_height, width, height, start_x_min, start_y_min, start_x_max, start_y_max, i, j)

                h_start = i * 1.0 / (image_height - height + 1e-10)
                w_start = j * 1.0 / (image_width - width + 1e-10)

                crop_coords = fcrops.get_crop_coords(
                    (image_height, image_width), (height, width), h_start, w_start
                )

                return {"crop_coords": crop_coords}

        # Fallback to whole image (full) crop
        if self.crop_size is not None:
            height, width = self.crop_size
            start_x = random.randint(0, max(0, image_width - width))
            start_y = random.randint(0, max(0, image_height - height))
            end_x = min(start_x + width, image_width)
            end_y = min(start_y + height, image_height)
            crop_coords = [start_x, start_y, end_x, end_y]
        else:
            crop_coords = [0, 0, image_width, image_height]
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "scale", "ratio", "get_bbox_func", "retry"


class CustomCoarseDropout(A.CoarseDropout):
    def apply_to_keypoints(self, keypoints, **kwargs):
        # print('coarse dropout keypoints')
        # just no change
        return keypoints


# @BUGFIX: for albumentations 1.4.16 (24/09/2024)
# incorrect order of width/height in several class methods
class CustomGridDropout(A.GridDropout):
    _targets_ = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        ratio: float = Field(
            description="The ratio of the mask holes to the unit_size.", gt=0, le=1
        )

        unit_size_min: int | None = Field(
            None, description="Minimum size of the grid unit.", ge=2
        )
        unit_size_max: int | None = Field(
            None, description="Maximum size of the grid unit.", ge=2
        )

        holes_number_x: int | None = Field(
            None, description="The number of grid units in x direction.", ge=1
        )
        holes_number_y: int | None = Field(
            None, description="The number of grid units in y direction.", ge=1
        )

        shift_x: int | None = Field(
            0, description="Offsets of the grid start in x direction.", ge=0
        )
        shift_y: int | None = Field(
            0, description="Offsets of the grid start in y direction.", ge=0
        )

        random_offset: bool = Field(
            False, description="Whether to offset the grid randomly."
        )
        fill_value: ColorType | None = Field(
            0, description="Value for the dropped pixels."
        )
        mask_fill_value: ColorType | None = Field(
            None, description="Value for the dropped pixels in mask."
        )
        unit_size_range: (
            Annotated[
                tuple[int, int],
                AfterValidator(check_1plus),
                AfterValidator(nondecreasing),
            ]
            | None
        ) = None
        shift_xy: Annotated[tuple[int, int], AfterValidator(check_0plus)] = Field(
            (0, 0),
            description="Offsets of the grid start in x and y directions.",
        )
        holes_number_xy: tuple[tuple[int, int], tuple[int, int]] | None = Field(
            None,
            description="(Customized, no check) The number of grid units in x and y directions.",
        )

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: int | None = None,
        unit_size_max: int | None = None,
        holes_number_x: int | None = None,
        holes_number_y: int | None = None,
        shift_x: int | None = None,
        shift_y: int | None = None,
        random_offset: bool = False,
        fill_value: ColorType = 0,
        mask_fill_value: ColorType | None = None,
        unit_size_range: tuple[int, int] | None = None,
        holes_number_xy: tuple[tuple[int, int], tuple[int, int]] | None = None,
        shift_xy: tuple[int, int] = (0, 0),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        DualTransform.__init__(self, p, always_apply)
        self.ratio = ratio
        self.unit_size_range = unit_size_range
        self.holes_number_xy = holes_number_xy
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        self.shift_xy = shift_xy

    def _calculate_dimensions_based_on_holes(self, shape) -> tuple[int, int]:
        """Calculates dimensions based on the number of holes specified."""
        height, width = shape[:2]
        if self.holes_number_xy is not None:
            holes_number_x, holes_number_y = self.holes_number_xy or (None, None)
            holes_number_x = random.randint(holes_number_x[0], holes_number_x[1])
            holes_number_y = random.randint(holes_number_y[0], holes_number_y[1])
        else:
            holes_number_x, holes_number_y = None, None
        unit_width = self._calculate_dimension(width, holes_number_x, 10)
        unit_height = self._calculate_dimension(height, holes_number_y, unit_width)
        # print(unit_width, unit_height, width, height, holes_number_y, holes_number_x)
        return unit_width, unit_height

    def _calculate_hole_dimensions(
        self, unit_shape: tuple[int, int]
    ) -> tuple[int, int]:
        """Calculates the dimensions of the holes to be dropped out."""
        unit_width, unit_height = unit_shape
        hole_width = min(max(1, int(unit_width * self.ratio)), unit_width - 1)
        hole_height = min(max(1, int(unit_height * self.ratio)), unit_height - 1)
        return hole_width, hole_height

    @staticmethod
    def _generate_holes(
        image_shape: tuple[int, int],
        unit_shape: tuple[int, int],
        hole_dimensions: tuple[int, int],
        shift_x: int,
        shift_y: int,
    ) -> np.ndarray:
        height, width = image_shape[:2]
        unit_width, unit_height = unit_shape
        hole_width, hole_height = hole_dimensions
        """Generates the list of holes to be dropped out."""
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                x1 = min(shift_x + unit_width * i, width)
                y1 = min(shift_y + unit_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))
        return np.array(holes)

    def apply_to_keypoints(self, keypoints, **kwargs):
        # just no change
        # print('grid dropout keypoints')
        return keypoints


class CustomXYMasking(A.XYMasking):
    class InitSchema(BaseTransformInitSchema):
        num_masks_x: NonNegativeIntRangeType
        num_masks_y: NonNegativeIntRangeType
        mask_x_length: tuple[float, float]
        mask_y_length: tuple[float, float]

        fill_value: ColorType
        mask_fill_value: ColorType

        @model_validator(mode="after")
        def check_mask_length(self) -> Self:
            return self

    def __init__(
        self,
        num_masks_x: ScaleIntType = 0,
        num_masks_y: ScaleIntType = 0,
        mask_x_length: ScaleIntType = 0,
        mask_y_length: ScaleIntType = 0,
        fill_value: ColorType = 0,
        mask_fill_value: ColorType = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        DualTransform.__init__(self, p, always_apply)
        self.num_masks_x = cast(Tuple[int, int], num_masks_x)
        self.num_masks_y = cast(Tuple[int, int], num_masks_y)

        self.mask_x_length = cast(Tuple[float, float], mask_x_length)
        self.mask_y_length = cast(Tuple[float, float], mask_y_length)
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        height, width = params["shape"][:2]

        mask_x_length = [max(1, int(e * width)) for e in self.mask_x_length]
        mask_y_length = [max(1, int(e * height)) for e in self.mask_y_length]

        # Use the helper method to validate mask lengths against image dimensions
        self.validate_mask_length(mask_x_length, width, "mask_x_length")
        self.validate_mask_length(mask_y_length, height, "mask_y_length")

        masks_x = self.generate_masks(
            self.num_masks_x, width, height, mask_x_length, axis="x"
        )
        masks_y = self.generate_masks(
            self.num_masks_y, width, height, mask_y_length, axis="y"
        )

        return {"masks_x": masks_x, "masks_y": masks_y}

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        # print('xy masking keypoints')
        return keypoints
