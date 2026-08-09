from figuregen.util import image
import figuregen as fig

from typing import Tuple, List
import numpy as np

from utils import relative_mse, SUPPORTED_ERROR_METRICS, CROP_COLORS, ORDER_COLORS_LATEX


class MyCropComparison:
    """ Matrix of cropped and zoomed images next to a reference image.

    Derived classes can change some behaviour, like the choice of error metric, by overriding the
    corresponding methods.

    Additional content can be added (or removed) by the user simply by accessing the individual
    grids in the generated list of grids.
    """

    def __init__(self, reference_image, method_images, crops: List[image.Cropbox],
                 scene_name=None, method_names=None, use_latex=False, error_metric="relMSE",
                 errors_provided: List[List[float]] = None, error_metrics_provided: List[str] = None, header=True, use_color=False):
        """ Shows a reference image next to a grid of crops from different methods.

        Args:
            reference_image: a reference image (or any other image to put full-size in the lefthand grid)
            method_images: list of images, each corresponds to a new column in the crop grid
            crops: list of crops to take from each method, each creates a new row and a marker on the reference, only for visualization, not used for error calculation
            scene_name: [optional] string, name of the scene to put underneath the reference image
            method_names: [optional] list of string, names for the reference and each method, to put above the crops
            use_latex: set to true to pretty-print captions with LaTeX commands (requires TikZ backend)
            error_metric: [optional] string, name of the error metric to use, defaults to "relMSE"

            errors_provided: [optional] np 2d array, first dimension is the metric, second the method
            error_metrics_provided: [optional] list of error metrics to use, if not given, uses relative MSE

            use_color: [optional] only effective when use_latex = True; if True, color the minimum two errors, if False, only bold the minimum error

        Returns:
            A list of two grids:
            The first is a single image (reference), the second a series of crops, one or more for each method.
        """

        # judge whether we should calculate errors or not
        should_calculate_errors = False
        if errors_provided is None:
            assert error_metrics_provided is None, "If you provide errors, you must also provide error metrics"
            should_calculate_errors = True

        self._reference_image = reference_image
        self._method_images = method_images
        self._use_latex = use_latex
        self._use_color = use_color if use_latex else False

        if should_calculate_errors:
            self._error_metric_names = [error_metric]
            self._errors = [[
                self.compute_error(reference_image, m)
                for m in method_images
            ]]
        else:
            assert len(error_metrics_provided) == len(errors_provided), "Error metrics and errors must have the same length"
            self._error_metric_names = error_metrics_provided
            self._errors = errors_provided

        # Create the grid for the reference image
        self._ref_grid = fig.Grid(1, 1)
        self._ref_grid[0, 0].image = self.tonemap(reference_image)
        for i in range(len(crops)):
            crop = crops[i]
            self._ref_grid[0, 0].set_marker(crop.marker_pos, crop.marker_size, color=CROP_COLORS[i % len(CROP_COLORS)])

        if scene_name is not None:
            scene_title = None
            # if (use_latex):
            # scene_title = "\\textsc{{{}}}".format(scene_name)
            # else:
            scene_title = scene_name
            self._ref_grid.set_col_titles(fig.BOTTOM, [scene_title])

        # Create the grid with the crops
        self._crop_grid = fig.Grid(num_cols=len(method_images) + 1, num_rows=len(crops))
        for row in range(len(crops)):
            self._crop_grid[row, 0].image = self.tonemap(crops[row].crop(reference_image))
            self._crop_grid[row, 0].set_frame(linewidth=1, color=CROP_COLORS[row % len(CROP_COLORS)])

            for col in range(len(method_images)):
                self._crop_grid[row, col + 1].image = self.tonemap(crops[row].crop(method_images[col]))
                self._crop_grid[row, col + 1].set_frame(linewidth=1, color=CROP_COLORS[row % len(CROP_COLORS)])

        # Put error values underneath the columns
        join_str = "\\\\" if use_latex else "\n"
        metric_num = len(self._error_metric_names)
        # colorbox set the background color; fboxsep sets the size of the colorbox
        error_strings_formatter = ("{{\\fboxsep1pt\\colorbox{{white}}{{{}}}}}") if self._use_color else "{}"
        error_strings = [[error_strings_formatter.format(i)] for i in self._error_metric_names]
        for i in range(metric_num):
            # i=0 RelMSE; i=1 RayEffInv
            error_strings[i].extend([self.error_string(j, self._errors[i], i != 0) for j in range(len(self._errors[i]))])
        transpose_error_strings = [list(i) for i in zip(*error_strings)]
        error_strings = [join_str.join(i) for i in transpose_error_strings]
        self._crop_grid.set_col_titles(fig.BOTTOM, error_strings)

        bottom_height_scale = 4.0 if self._use_color else 3.5

        crop_layout = self._crop_grid.layout
        crop_layout.row_space = 1
        crop_layout.column_space = 1
        crop_layout.column_titles[fig.BOTTOM] = fig.TextFieldLayout(fontsize=8, size=bottom_height_scale * metric_num, offset=0.5)

        # If given, show method names on top
        if method_names is not None and header:
            self._crop_grid.set_col_titles(fig.TOP, method_names)
            crop_layout.column_titles[fig.TOP] = fig.TextFieldLayout(fontsize=8, size=2.8, offset=0.25)

        self._ref_grid.copy_layout(self._crop_grid)
        self._ref_grid.layout.column_titles[fig.BOTTOM] = fig.TextFieldLayout(fontsize=12, size=bottom_height_scale * metric_num, offset=0.5)
        self._ref_grid.layout.padding[fig.RIGHT] = 1

    def tonemap(self, img):
        return fig.JPEG(image.lin_to_srgb(img), quality=80)

    def error_metric_names(self) -> List[str]:
        """ Returns the names of the error metrics used in this figure. """
        return self._error_metric_names

    def compute_error(self, reference_image, method_image) -> Tuple[str, List[float]]:
        # here, if we call this, only one metric is used, so we can just return the first one
        metric = self._error_metric_names[0]
        if metric == "relMSE":
            return relative_mse(method_image, reference_image)
        else:
            raise ValueError("Unknown error metric: {}. Supported: {}".format(metric, SUPPORTED_ERROR_METRICS))

    def error_string(self, index: int, errors: List[float], clamped2int: bool = False) -> str:
        """ Generates the human-readable error string for the i-th element in a list of error values.

        Args:
            index: index in the list of errors
            errors: list of error values, one per method, in order
        """
        e = errors[index]
        e0 = errors[0]  # refs

        float_fmt = ".0f" if clamped2int else ".2f"

        if self._use_latex:
            if (self._use_color):
                rank = sum(errors[i] < e for i in range(len(errors)))
                rank = min(rank, len(ORDER_COLORS_LATEX) - 1)
                return ("{{\\fboxsep1pt\\colorbox{}{{${:{fmt}}\\;({:.2f}\\times)$}}}}"
                        .format(ORDER_COLORS_LATEX[rank], e, e / e0, fmt=float_fmt))
            else:
                # bold the minimum error
                bold_prefix = "\\mathbf" if index == np.argmin(errors) else ""
                return ("${}{{{{:{fmt}}}\\;({:.2f}\\times)}}$"
                        .format(bold_prefix, e, e / e0, fmt=float_fmt))
        else:
            return "{:{fmt}} ({:.2f}x)".format(e, e / e0, fmt=float_fmt)

    @property
    def figure_row(self) -> List[fig.Grid]:
        return [self._ref_grid, self._crop_grid]
