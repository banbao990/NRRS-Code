"""Export experiment EXRs and their final metrics.

Input files follow the convention used by ``main_exps.py``::

    <image_dir>/<scene>_<method>.exr
    <image_dir>/<scene>_<method>.exr.json

Edit ``SCENES`` and ``METHODS`` below to select the inputs. Every method is
exported for every scene. For example::

    SCENES = ["apartment1", "sponza-caustic"]
    METHODS = ["pt", "nrrs+", "1#0@nrrs+"]

Then run ``python export_pngs.py --image-dir <experiment-directory>``. Jobs are
processed by a thread pool, and an existing valid PNG is reused unless
``--force`` is supplied.

Results are written to ``<image_dir>/pngs/<scene>/``. Each scene directory
contains one ``<method>.png`` per selected method, ``reference.png``,
``metrics.json``, and a static comparison page. Reference images are displayed
but are not counted as methods. Open ``<image_dir>/pngs/index.html`` to browse.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import re
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

# OpenCV checks this switch while it is imported, so it must be set first.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import json5
import numpy as np
from simpleimageio import lin_to_srgb

# Parallelize at the Python level instead of nesting OpenCV worker pools.
cv.setNumThreads(1)


# ------------------------------ Configuration ------------------------------
# Every method below is exported for every scene below.
SCENES = [
    # main
    "apartment1",
    "apartment3",
    "bistro-interior",
    "noenv_apartment2",
    "sponza-caustic",
    "sun-temple-v2-lighter-view2",

    # supplementary
    "apartment1-v2-view2",
    "m1-f210",
    "sponza-glossy",
    "sun-temple",
    "tropical_bedroom_cx",
    "san-miguel-view2",
]

METHODS = [
    "pt",
    "adrrs",
    "ears",
    "adn",
    "nrrs+",
    "1#0@nrrs+",
]

# 8 workers keep memory usage reasonable while processing HDR float images.
MAX_WORKERS = min(8, os.cpu_count() or 1)
# ---------------------------------------------------------------------------


SCENE_TITLE_OVERRIDES = {
    "apartment1-v2": "pool",
    "apartment1-v2-view2": "pool-large",
    "m1-f210": "measure-one",
    "noenv_apartment2": "apartment2",
    "sponza-view2": "sponza",
    "sun-temple-v2": "sun-temple-dark",
    "sun-temple-v2-view2": "sun-temple-interior",
    "sun-temple-v2-lighter-view2": "sun-temple-interior",
    "tropical_bedroom_cx": "tropical-bedroom",
    "cornell-box2": "cornell-box",
    "cylinder-2": "cylinder",
    "sponza-caustic-finer3-v2-view2": "sponza-caustic-finer",
    "san-miguel-view2": "san-miguel",
    "sponza-caustic-lowf": "sponza-caustic (smooth)",
}

METHOD_TITLES = {
    "pt": "PT",
    "adrrs": "ADRRS",
    "ears": "EARS",
    "ears+": "EARS+",
    "adn": "ADRRS (NN)",
    "nrrs": "NRRS",
    "nrrs+": "NRRS+",
    "1#0@nrrs": "AID-NRRS",
    "1#0@nrrs+": "AID-NRRS+",
}


class ExportError(RuntimeError):
    """An expected input or export error with a user-facing message."""


@dataclass(frozen=True)
class ExportResult:
    scene: str
    method: str
    metrics: dict[str, float]
    width: int
    height: int
    generated: bool


@dataclass(frozen=True)
class ReferenceResult:
    scene: str
    width: int
    height: int
    generated: bool


_CROP_FIELDS = ("top", "left", "height", "width")


def _literal_string(node: ast.AST, context: str) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ExportError(f"Expected a string in {context}") from exc
    if not isinstance(value, str):
        raise ExportError(f"Expected a string in {context}")
    return value


def _crop_dict_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "CROPS_DICT":
        return None
    try:
        value = ast.literal_eval(node.slice)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _crop_number(node: ast.AST, field: str, context: str) -> int:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ExportError(f"Crop {field} must be an integer in {context}") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"Crop {field} must be an integer in {context}")
    return value


def _parse_cropbox(node: ast.AST, context: str) -> dict[str, int]:
    if not isinstance(node, ast.Call):
        raise ExportError(f"Expected Cropbox(...) in {context}")
    function_name = (
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else ""
    )
    if function_name != "Cropbox":
        raise ExportError(f"Expected Cropbox(...) in {context}")

    values: dict[str, int] = {}
    for field, argument in zip(_CROP_FIELDS, node.args):
        values[field] = _crop_number(argument, field, context)
    for keyword in node.keywords:
        if keyword.arg in _CROP_FIELDS:
            values[keyword.arg] = _crop_number(keyword.value, keyword.arg, context)

    missing = [field for field in _CROP_FIELDS if field not in values]
    if missing:
        raise ExportError(f"Cropbox is missing {', '.join(missing)} in {context}")
    if values["top"] < 0 or values["left"] < 0:
        raise ExportError(f"Crop top and left must be non-negative in {context}")
    if values["height"] <= 0 or values["width"] <= 0:
        raise ExportError(f"Crop height and width must be positive in {context}")
    return values


def _parse_crop_list(node: ast.AST, context: str) -> list[dict[str, int]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ExportError(f"Expected a list of Cropbox values in {context}")
    return [
        _parse_cropbox(crop, f"{context} crop {index + 1}")
        for index, crop in enumerate(node.elts)
    ]


def load_scene_crops(
    source_path: Path, scenes: Sequence[str]
) -> dict[str, list[dict[str, int]]]:
    """Read CROPS_DICT from main_exps.py without importing its heavy dependencies."""

    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Cannot read crop definitions from {source_path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        raise ExportError(f"Cannot parse crop definitions in {source_path}: {exc}") from exc

    crop_sets: dict[str, list[dict[str, int]]] = {}
    aliases: dict[str, str] = {}
    found_dictionary = False

    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        context = f"{source_path}:{statement.lineno}"
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == "CROPS_DICT":
                found_dictionary = True
                if not isinstance(statement.value, ast.Dict):
                    raise ExportError(f"CROPS_DICT must be a dictionary in {context}")
                for scene_node, crops_node in zip(
                    statement.value.keys, statement.value.values
                ):
                    if scene_node is None:
                        raise ExportError(f"CROPS_DICT unpacking is not supported in {context}")
                    scene = _literal_string(scene_node, context)
                    crop_sets[scene] = _parse_crop_list(crops_node, f"{context} [{scene}]")
                continue

            target_scene = _crop_dict_key(target)
            if target_scene is None:
                continue
            source_scene = _crop_dict_key(statement.value)
            if source_scene is not None:
                aliases[target_scene] = source_scene
            else:
                crop_sets[target_scene] = _parse_crop_list(
                    statement.value, f"{context} [{target_scene}]"
                )

    if not found_dictionary:
        raise ExportError(f"CROPS_DICT was not found in {source_path}")

    unresolved = dict(aliases)
    while unresolved:
        resolved_any = False
        for target_scene, source_scene in list(unresolved.items()):
            if source_scene not in crop_sets:
                continue
            crop_sets[target_scene] = [dict(crop) for crop in crop_sets[source_scene]]
            del unresolved[target_scene]
            resolved_any = True
        if not resolved_any:
            descriptions = ", ".join(
                f"{target} -> {source}" for target, source in unresolved.items()
            )
            raise ExportError(f"Cannot resolve CROPS_DICT aliases: {descriptions}")

    selected_crops: dict[str, list[dict[str, int]]] = {}
    for scene in scenes:
        crops = crop_sets.get(scene)
        if crops is None:
            raise ExportError(f"No CROPS_DICT entry exists for scene {scene!r}")
        if len(crops) != 2:
            raise ExportError(
                f"Scene {scene!r} must have exactly two crops; found {len(crops)}"
            )
        selected_crops[scene] = [dict(crop) for crop in crops]
    return selected_crops


def _simple_filename(value: str, kind: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ExportError(f"Invalid {kind} name: {value!r}")
    return value


def selection_from_arrays(
    scenes: Sequence[str], methods: Sequence[str]
) -> dict[str, list[str]]:
    if not scenes:
        raise ExportError("SCENES must contain at least one scene")
    if not methods:
        raise ExportError("METHODS must contain at least one method")

    normalized_methods: list[str] = []
    for raw_method in methods:
        if not isinstance(raw_method, str):
            raise ExportError("METHODS may only contain strings")
        method = _simple_filename(raw_method, "method")
        if method not in normalized_methods:
            normalized_methods.append(method)

    selection: dict[str, list[str]] = {}
    for raw_scene in scenes:
        if not isinstance(raw_scene, str):
            raise ExportError("SCENES may only contain strings")
        scene = _simple_filename(raw_scene, "scene")
        selection[scene] = list(normalized_methods)
    return selection


def infer_reference_dir(image_dir: Path) -> Path:
    match = re.match(r"^d(\d+)(?:-|$)", image_dir.name, flags=re.IGNORECASE)
    if match is None:
        raise ExportError(
            "Cannot infer reference depth from the image directory name; "
            "pass --reference-dir explicitly"
        )

    project_dir = Path(__file__).resolve().parents[2]
    reference_dir = (
        project_dir
        / "common"
        / "assets"
        / "scenes-nrrs"
        / "references"
        / f"d{match.group(1)}"
    )
    if not reference_dir.is_dir():
        raise ExportError(f"Reference directory does not exist: {reference_dir}")
    return reference_dir


def find_reference_exr(reference_dir: Path, scene: str) -> Path:
    preferred = reference_dir / f"{scene}_200000.exr"
    if preferred.is_file():
        return preferred

    prefix = f"{scene}_"
    candidates: list[tuple[int, Path]] = []
    for path in reference_dir.glob(f"{scene}_*.exr"):
        sample_text = path.stem[len(prefix):]
        if sample_text.isdigit():
            candidates.append((int(sample_text), path))

    if not candidates:
        raise ExportError(f"Reference EXR not found for scene {scene!r} in {reference_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def build_jobs(image_dir: Path, selection: Mapping[str, Sequence[str]]) -> dict[str, list[tuple[str, Path, Path]]]:
    jobs: dict[str, list[tuple[str, Path, Path]]] = {}
    missing: list[Path] = []

    for scene, methods in selection.items():
        scene_jobs: list[tuple[str, Path, Path]] = []
        for method in methods:
            exr_path = image_dir / f"{scene}_{method}.exr"
            sidecar_path = Path(f"{exr_path}.json")
            if not exr_path.is_file():
                missing.append(exr_path)
            if not sidecar_path.is_file():
                missing.append(sidecar_path)
            scene_jobs.append((method, exr_path, sidecar_path))
        jobs[scene] = scene_jobs

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise ExportError(f"Missing input file(s):\n{details}")
    return jobs


def read_exr(exr_path: Path) -> np.ndarray:
    """Read linear RGB using the same channel convention as main_exps.py."""
    try:
        image = cv.imread(str(exr_path), cv.IMREAD_UNCHANGED)
    except cv.error as exc:
        raise ExportError(f"OpenCV could not read EXR image {exr_path}: {exc}") from exc
    if image is None:
        raise ExportError(f"OpenCV could not read EXR image: {exr_path}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ExportError(
            f"Expected at least three channels in {exr_path}, got shape {image.shape}"
        )

    # OpenCV returns BGR(A); downstream conversion works in RGB.
    return np.asarray(image[..., :3][..., ::-1], dtype=np.float32)


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    """Read width and height directly from a PNG IHDR without decoding it."""
    try:
        with png_path.open("rb") as png_file:
            header = png_file.read(24)
    except OSError as exc:
        raise ExportError(f"Cannot read existing PNG {png_path}: {exc}") from exc

    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ExportError(f"Existing PNG has an invalid header: {png_path}")

    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ExportError(f"Existing PNG has invalid dimensions: {png_path}")
    return width, height


def write_png(linear_rgb: np.ndarray, png_path: Path) -> None:
    """Tone-map linear RGB and atomically write an 8-bit sRGB PNG."""
    finite_rgb = np.nan_to_num(linear_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    srgb = np.clip(lin_to_srgb(finite_rgb), 0.0, 1.0)
    rgb8 = np.rint(srgb * 255.0).astype(np.uint8)

    temporary_path = png_path.with_name(
        f".{png_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.png"
    )
    try:
        # cv.imwrite expects BGR rather than RGB.
        written = cv.imwrite(str(temporary_path), rgb8[..., ::-1])
        if not written:
            raise ExportError(f"OpenCV could not write PNG image: {png_path}")
        temporary_path.replace(png_path)
    except cv.error as exc:
        raise ExportError(f"OpenCV could not write PNG image {png_path}: {exc}") from exc
    except OSError as exc:
        raise ExportError(f"Cannot finalize PNG image {png_path}: {exc}") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_metrics(sidecar_path: Path, pixel_count: int) -> dict[str, float]:
    try:
        sidecar_text = sidecar_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Cannot read metrics file {sidecar_path}: {exc}") from exc

    try:
        # Experiment sidecars are normally strict JSON. The stdlib parser is
        # substantially faster than json5 for the long error-history arrays.
        data = json.loads(sidecar_text)
    except json.JSONDecodeError:
        try:
            data = json5.loads(sidecar_text)
        except ValueError as exc:
            raise ExportError(f"Invalid JSON/JSON5 metrics file {sidecar_path}") from exc

    try:
        relmse = float(data["data"][-1]["RelMSE"])
        rays = float(data["rays"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ExportError(
            f"Metrics file {sidecar_path} must contain data[-1].RelMSE and rays"
        ) from exc

    if not math.isfinite(relmse) or not math.isfinite(rays) or relmse < 0 or rays < 0:
        raise ExportError(f"Metrics file {sidecar_path} contains invalid RelMSE or rays")

    # main_exps.py calls this value RayEffInv. Keep the requested JSON key
    # "rayeff", but retain the same lower-is-better definition.
    rayeff = relmse * rays / pixel_count
    return {"relmse": relmse, "rayeff": rayeff}


def write_metrics(metrics_path: Path, metrics: Mapping[str, Mapping[str, float]]) -> None:
    temporary_path = metrics_path.with_suffix(".json.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
            metrics_file.write("\n")
        temporary_path.replace(metrics_path)
    except OSError as exc:
        raise ExportError(f"Cannot write metrics file {metrics_path}: {exc}") from exc


def write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
        temporary_path.replace(path)
    except OSError as exc:
        raise ExportError(f"Cannot write static viewer file {path}: {exc}") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def scene_title(scene: str) -> str:
    mapped = SCENE_TITLE_OVERRIDES.get(scene, scene)
    return mapped.replace("_", " ").replace("-", " ").title()


def method_title(method: str) -> str:
    return METHOD_TITLES.get(method, method.replace("_", " ").title())


def process_job(
    scene: str,
    method: str,
    exr_path: Path,
    sidecar_path: Path,
    output_root: Path,
    fixed_pixel_count: int | None,
    force: bool,
) -> ExportResult:
    png_path = output_root / scene / f"{method}.png"

    if png_path.is_file() and not force:
        try:
            width, height = read_png_dimensions(png_path)
        except ExportError:
            # A truncated/invalid PNG should not poison all future resume runs.
            pass
        else:
            pixel_count = fixed_pixel_count or (width * height)
            metrics = read_metrics(sidecar_path, pixel_count)
            return ExportResult(scene, method, metrics, width, height, False)

    image = read_exr(exr_path)
    height, width = image.shape[:2]
    pixel_count = fixed_pixel_count or (width * height)
    metrics = read_metrics(sidecar_path, pixel_count)
    write_png(image, png_path)
    return ExportResult(scene, method, metrics, width, height, True)


def process_reference_job(
    scene: str,
    reference_path: Path,
    output_root: Path,
    force: bool,
) -> ReferenceResult:
    png_path = output_root / scene / "reference.png"
    if png_path.is_file() and not force:
        try:
            width, height = read_png_dimensions(png_path)
        except ExportError:
            pass
        else:
            return ReferenceResult(scene, width, height, False)

    image = read_exr(reference_path)
    height, width = image.shape[:2]
    write_png(image, png_path)
    return ReferenceResult(scene, width, height, True)


def generate_viewer(
    output_root: Path,
    selection: Mapping[str, Sequence[str]],
    metrics_by_scene: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Path:
    crops_by_scene = load_scene_crops(
        Path(__file__).with_name("main_exps.py"), tuple(selection)
    )
    source_assets = Path(__file__).with_name("image_viewer")
    output_assets = output_root / "assets"
    try:
        output_assets.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"Cannot create viewer asset directory {output_assets}: {exc}") from exc

    for asset_name in ("viewer.css", "viewer.js"):
        source_path = source_assets / asset_name
        try:
            content = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Cannot read viewer asset {source_path}: {exc}") from exc
        write_text(output_assets / asset_name, content)

    cards: list[str] = []
    overview_scenes = sorted(
        selection.items(),
        key=lambda item: scene_title(item[0]).casefold(),
    )
    for scene, methods in overview_scenes:
        title = scene_title(scene)
        best_method = min(methods, key=lambda method: metrics_by_scene[scene][method]["relmse"])
        scene_url = quote(scene, safe="")
        thumbnail_url = quote("reference.png", safe="")
        method_names = " · ".join(method_title(method) for method in methods)
        cards.append(
            f"""
            <a class="scene-card" href="{scene_url}/">
                <div class="scene-card__image">
                    <img src="{scene_url}/{thumbnail_url}" alt="{html.escape(title)} preview" loading="lazy">
                    <span class="scene-card__count">{len(methods)} methods</span>
                </div>
                <div class="scene-card__body">
                    <h2>{html.escape(title)}</h2>
                    <p>{html.escape(method_names)}</p>
                    <span class="scene-card__best">Best RelMSE · {html.escape(method_title(best_method))}</span>
                </div>
            </a>"""
        )

    overview_html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Interactive experiment result viewer">
    <title>Interactive Result Viewer</title>
    <link rel="shortcut icon" href="/utils/img/banbao(990).jpg" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&amp;display=swap">
    <link rel="stylesheet" href="assets/viewer.css">
</head>
<body class="overview-page">
    <header class="overview-header">
        <p class="eyebrow">NRRS experiment results</p>
        <h1>Interactive Result Viewer</h1>
        <p class="overview-lead">Select a scene to compare every configured method at full resolution (720p).</p>
        <p class="overview-note">All images use 60 seconds of rendering time. Methods that require training use 60 seconds of training time.</p>
        <div class="overview-actions">
            <div class="overview-summary"><strong>{len(selection)}</strong> scenes <span></span> <strong>{len(next(iter(selection.values())))}</strong> methods per scene</div>
            <a class="back-link overview-home-link" href="../">← Homepage</a>
        </div>
    </header>
    <main class="scene-grid">
        {''.join(cards)}
    </main>
    <footer class="site-footer">Interactive Result Viewer for NRRS.</footer>
</body>
</html>
"""
    write_text(output_root / "index.html", overview_html)

    for scene, methods in selection.items():
        title = scene_title(scene)
        method_data = [
            {
                "id": method,
                "title": method_title(method),
                "image": quote(f"{method}.png", safe=""),
                "relmse": metrics_by_scene[scene][method]["relmse"],
                "rayeff": metrics_by_scene[scene][method]["rayeff"],
            }
            for method in methods
        ]
        method_data.append(
            {
                "id": "reference",
                "title": "Reference",
                "image": quote("reference.png", safe=""),
                "relmse": None,
                "rayeff": None,
                "isReference": True,
            }
        )
        viewer_data = {
            "scene": scene,
            "title": title,
            "crops": crops_by_scene[scene],
            "methods": method_data,
        }
        scene_dir = output_root / scene
        write_text(
            scene_dir / "data.js",
            "window.VIEWER_DATA = "
            + json.dumps(viewer_data, ensure_ascii=False, indent=2)
            + ";\n",
        )

        scene_html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Interactive comparison for {html.escape(title)}">
    <title>{html.escape(title)} · Result Viewer</title>
    <link rel="shortcut icon" href="/utils/img/banbao(990).jpg" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&amp;display=swap">
    <link rel="stylesheet" href="../assets/viewer.css">
</head>
<body class="viewer-page">
    <header class="viewer-header viewer-shell">
        <a class="back-link" href="../" aria-label="Back to all scenes">← All scenes</a>
        <div>
            <p class="eyebrow">Interactive comparison</p>
            <h1>{html.escape(title)}</h1>
        </div>
        <p class="viewer-help">Wheel to zoom · drag to pan · number keys to switch · R to reset</p>
    </header>
    <main class="viewer-shell">
        <div class="viewer-grid">
            <section class="viewer-panel" aria-label="Image comparison">
                <div class="viewer-toolbar">
                    <div class="toolbar-copy">
                        <span class="toolbar-label">Active method</span>
                        <span id="method-title"></span>
                    </div>
                    <div class="toolbar-actions">
                        <div id="crop-actions" class="crop-actions" aria-label="Crop presets"></div>
                        <span id="zoom-readout">100%</span>
                        <button id="reset-view" type="button">Reset view</button>
                    </div>
                </div>
                <div id="image-stage" tabindex="0" aria-label="Zoomable result image">
                    <img id="main-image" alt="" draggable="false">
                    <div class="stage-overlay"><span>Full-resolution PNG · 720p</span></div>
                </div>
                <div id="method-tabs" role="tablist" aria-label="Rendering methods"></div>
            </section>
            <aside class="side-panel">
                <div>
                    <p class="eyebrow">Current result</p>
                    <h2 class="panel-title">Error metrics</h2>
                    <p class="panel-note">Lower values are better.</p>
                </div>
                <div class="metric-cards" aria-live="polite">
                    <div class="metric-card">
                        <span class="metric-name">RelMSE (×10<sup>−3</sup>)</span>
                        <strong id="metric-relmse" class="metric-value"></strong>
                    </div>
                    <div class="metric-card">
                        <span class="metric-name">RayEffInv</span>
                        <strong id="metric-rayeff" class="metric-value"></strong>
                    </div>
                </div>
                <section class="side-metrics" aria-labelledby="metrics-heading">
                    <p class="eyebrow">Numerical comparison</p>
                    <h2 id="metrics-heading" class="panel-title">All metrics</h2>
                    <p class="panel-note">Lower values are better.</p>
                    <div class="table-scroll">
                        <table>
                            <thead><tr><th>Method</th><th>RelMSE (×10<sup>−3</sup>)</th><th>RayEffInv</th></tr></thead>
                            <tbody id="metrics-body"></tbody>
                        </table>
                    </div>
                </section>
                <div class="shortcut-list" aria-label="Viewer controls">
                    <div class="shortcut-row"><span>Switch image</span><kbd>1–{min(len(methods) + 1, 9)}</kbd></div>
                    <div class="shortcut-row"><span>Zoom</span><kbd>Wheel</kbd></div>
                    <div class="shortcut-row"><span>Reset</span><kbd>R</kbd></div>
                </div>
            </aside>
        </div>
        <section id="detail-panel" class="detail-panel" aria-labelledby="detail-heading" hidden>
            <div class="detail-panel__header">
                <div>
                    <p class="eyebrow">Synchronized inspection</p>
                    <h2 id="detail-heading" class="panel-title">Live detail comparison</h2>
                    <p class="panel-note">Move over the main image to inspect the same source region across every result.</p>
                </div>
                <p id="detail-position" class="detail-panel__position"></p>
            </div>
            <div id="detail-strip" class="detail-strip" aria-label="Synchronized method details"></div>
        </section>
        <div class="lower-grid">
            <section class="thumbnail-panel" aria-labelledby="thumbnails-heading">
                <p class="eyebrow">Quick switch</p>
                <h2 id="thumbnails-heading" class="panel-title">Methods &amp; reference</h2>
                <div id="thumbnail-strip" aria-label="Method thumbnails"></div>
            </section>
        </div>
    </main>
    <footer class="site-footer">{html.escape(title)} · Full-resolution static comparison</footer>
    <script src="data.js"></script>
    <script src="../assets/viewer.js"></script>
</body>
</html>
"""
        write_text(scene_dir / "index.html", scene_html)

    return output_root / "index.html"


def export(
    image_dir: Path,
    reference_dir: Path,
    selection: Mapping[str, Sequence[str]],
    fixed_pixel_count: int | None,
    workers: int,
    force: bool,
) -> tuple[int, int, int, int, Path, Path]:
    if not image_dir.is_dir():
        raise ExportError(f"Image directory does not exist: {image_dir}")

    jobs = build_jobs(image_dir, selection)
    reference_jobs = {
        scene: find_reference_exr(reference_dir, scene) for scene in selection
    }
    output_root = image_dir / "pngs"
    for scene in selection:
        output_dir = output_root / scene
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(f"Cannot create output directory {output_dir}: {exc}") from exc

    results: dict[str, dict[str, ExportResult]] = {scene: {} for scene in selection}
    failures: list[str] = []
    generated = 0
    skipped = 0
    references_generated = 0
    references_skipped = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exr-export") as executor:
        future_context = {}
        for scene, scene_jobs in jobs.items():
            for method, exr_path, sidecar_path in scene_jobs:
                future = executor.submit(
                    process_job,
                    scene,
                    method,
                    exr_path,
                    sidecar_path,
                    output_root,
                    fixed_pixel_count,
                    force,
                )
                future_context[future] = ("method", scene, method)

        for scene, reference_path in reference_jobs.items():
            future = executor.submit(
                process_reference_job,
                scene,
                reference_path,
                output_root,
                force,
            )
            future_context[future] = ("reference", scene, "Reference")

        for future in as_completed(future_context):
            kind, scene, label = future_context[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(f"[{scene}] {label}: {exc}")
                print(f"[failed] [{scene}] {label}", file=sys.stderr)
                continue

            if kind == "reference":
                if result.generated:
                    references_generated += 1
                    status = "generated"
                else:
                    references_skipped += 1
                    status = "skipped"
                print(f"[{status}] [{scene}] Reference: {result.width}x{result.height}")
            else:
                results[scene][label] = result
                if result.generated:
                    generated += 1
                    status = "generated"
                else:
                    skipped += 1
                    status = "skipped"
                print(f"[{status}] [{scene}] {label}: {result.width}x{result.height}")

    metrics_by_scene: dict[str, dict[str, dict[str, float]]] = {}
    for scene, methods in selection.items():
        if any(method not in results[scene] for method in methods):
            continue
        ordered_metrics = {method: results[scene][method].metrics for method in methods}
        write_metrics(output_root / scene / "metrics.json", ordered_metrics)
        metrics_by_scene[scene] = ordered_metrics

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise ExportError(f"Failed to export {len(failures)} image(s):\n{details}")

    viewer_index = generate_viewer(output_root, selection, metrics_by_scene)
    return (
        generated,
        skipped,
        references_generated,
        references_skipped,
        output_root,
        viewer_index,
    )


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the SCENES x METHODS experiment EXRs configured in this file "
            "to PNG and export RelMSE/RayEff metrics."
        )
    )
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        required=True,
        type=Path,
        help="Directory containing <scene>_<method>.exr and sidecar JSON files.",
    )
    parser.add_argument(
        "--reference-dir",
        "--reference_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <scene>_<sample-count>.exr references. "
            "By default it is inferred from an image directory named like d6-final."
        ),
    )
    parser.add_argument(
        "--ray-eff-pixel-count",
        "--ray_eff_pixel_count",
        type=positive_integer,
        default=None,
        help=(
            "Fixed pixel count used to normalize RayEff. By default each EXR's actual "
            "width*height is used; pass 921600 to reproduce main_exps.py exactly."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=MAX_WORKERS,
        help=f"Number of parallel image workers (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate PNG files even when they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selection = selection_from_arrays(SCENES, METHODS)
        image_dir = args.image_dir.resolve()
        reference_dir = (
            args.reference_dir.resolve()
            if args.reference_dir is not None
            else infer_reference_dir(image_dir)
        )
        (
            generated,
            skipped,
            references_generated,
            references_skipped,
            output_root,
            viewer_index,
        ) = export(
            image_dir,
            reference_dir,
            selection,
            args.ray_eff_pixel_count,
            args.workers,
            args.force,
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {generated} image(s), skipped {skipped} existing image(s).")
    print(
        f"Generated {references_generated} reference(s), "
        f"skipped {references_skipped} existing reference(s)."
    )
    print(f"PNG directory: {output_root}")
    print(f"Static viewer: {viewer_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python scripts/exps/export_pngs.py --image-dir scripts/exps/images/exps/d6-final
