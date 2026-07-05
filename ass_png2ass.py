from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tempfile
from pathlib import Path


__version__ = "1.1.0"


class ConversionError(RuntimeError):
    pass


def load_rgba(path: Path, max_pixels: int) -> tuple[np.ndarray, int, int]:
    import numpy as np
    from PIL import Image

    if not path.exists():
        raise ConversionError(f"PNG not found: {path}")
    if not path.is_file():
        raise ConversionError(f"Input is not a file: {path}")

    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise ConversionError(f"Could not read image: {exc}") from exc

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ConversionError("Image has invalid dimensions.")
    if width * height > max_pixels:
        raise ConversionError(
            f"Image is too large ({width}x{height}). Limit is {max_pixels} pixels."
        )

    return np.array(image.convert("RGBA")), width, height


def threshold_to_byte(percent: float) -> int:
    percent = max(0.0, min(100.0, percent))
    return int(round(255.0 * percent / 100.0))


def detect_mode(rgba: np.ndarray, threshold: float) -> tuple[str, list[str]]:
    import numpy as np

    alpha = rgba[:, :, 3]
    if int(alpha.min()) < 255:
        return "alpha", ["auto mode selected alpha."]

    rgb = rgba[:, :, :3].astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    threshold_byte = threshold_to_byte(threshold)
    low_cut = min(threshold_byte, 64)
    high_cut = max(threshold_byte, 191)
    dark_count = int(np.count_nonzero(gray <= low_cut))
    bright_count = int(np.count_nonzero(gray >= high_cut))
    total = int(gray.size)

    if total > 0 and dark_count > 0 and bright_count > 0:
        binary_ratio = (dark_count + bright_count) / total
        if binary_ratio >= 0.90:
            if bright_count <= dark_count:
                return "white-matte", ["auto mode selected white-matte."]
            return "dark-matte", ["auto mode selected dark-matte."]

    return "color", ["auto mode selected color."]


def build_mask(rgba: np.ndarray, mode: str, threshold: float) -> tuple[np.ndarray, list[str], str]:
    import numpy as np

    threshold_byte = threshold_to_byte(threshold)
    warnings: list[str] = []

    if mode == "auto":
        mode, auto_warnings = detect_mode(rgba, threshold)
        warnings.extend(auto_warnings)

    if mode == "alpha":
        alpha = rgba[:, :, 3]
        if int(alpha.max()) == 255 and int(alpha.min()) == 255:
            raise ConversionError(
                "The image alpha is fully opaque. Use auto/white-matte/dark-matte/color mode or add transparency."
            )
        mask = alpha > max(0, threshold_byte)
    elif mode in {"luma", "dark-matte"}:
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        if np.count_nonzero(alpha < 1.0):
            rgb = rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
            warnings.append(f"{mode} mode composites transparency over white.")
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        mask = gray <= threshold_byte
    elif mode == "white-matte":
        rgb = rgba[:, :, :3].astype(np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        mask = gray >= threshold_byte
    elif mode == "color":
        alpha = rgba[:, :, 3]
        if int(alpha.min()) < 255:
            mask = alpha > 0
        else:
            mask = np.ones(alpha.shape, dtype=bool)
            warnings.append("color mode has no alpha; the whole canvas will be traced.")
    else:
        raise ConversionError(f"Unsupported mode: {mode}")

    return (mask.astype(np.uint8) * 255), warnings, mode


def trim_mask(mask: np.ndarray) -> tuple[tuple[int, int, int, int], list[str]]:
    import numpy as np

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ConversionError("The generated mask is empty.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    canvas_area = mask.shape[0] * mask.shape[1]
    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    visible_area = int(np.count_nonzero(mask))

    warnings: list[str] = []
    if bbox_area / canvas_area > 0.98 and visible_area / canvas_area > 0.90:
        warnings.append("Mask covers almost the whole canvas; this may trace the canvas.")

    return (x0, y0, x1, y1), warnings


def crop_for_trace(rgba: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], mode: str) -> Image.Image:
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = bbox
    cropped_rgba = rgba[y0 : y1 + 1, x0 : x1 + 1].copy()
    cropped_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    if mode in {"alpha", "luma", "dark-matte", "white-matte"}:
        out = np.zeros((cropped_mask.shape[0], cropped_mask.shape[1], 4), dtype=np.uint8)
        out[:, :, :3] = 255
        out[:, :, 3] = cropped_mask
        return Image.fromarray(out, "RGBA")

    cropped_rgba[:, :, 3] = np.minimum(cropped_rgba[:, :, 3], cropped_mask)
    return Image.fromarray(cropped_rgba, "RGBA")


def contour_to_ass(points: np.ndarray, coord_scale: int) -> str | None:
    if len(points) < 3:
        return None

    coords = []
    for point in points:
        x = int(round(float(point[0][0]) * coord_scale))
        y = int(round(float(point[0][1]) * coord_scale))
        coords.append((x, y))

    if len(coords) < 3:
        return None

    first_x, first_y = coords[0]
    parts = [f"m {first_x} {first_y}", "l"]
    parts.extend(f"{x} {y}" for x, y in coords[1:])
    if coords[-1] != coords[0]:
        parts.append(f"{first_x} {first_y}")
    return " ".join(parts)


def make_ass_text(shape: str, p_scale: int, pos_x: float, pos_y: float, blur: float) -> str:
    pos_x_i = int(round(pos_x))
    pos_y_i = int(round(pos_y))
    blur_tag = "" if blur <= 0 else f"\\blur{blur:g}"
    return f"{{\\an7\\pos({pos_x_i},{pos_y_i})\\bord0\\shad0{blur_tag}\\p{p_scale}}}{shape}{{\\p0}}"


def mask_to_opencv_lines(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    p_scale: int,
    simplify: float,
    min_area: float,
    pos_x: float,
    pos_y: float,
    blur: float,
) -> tuple[list[str], dict]:
    import cv2

    x0, y0, x1, y1 = bbox
    cropped = mask[y0 : y1 + 1, x0 : x1 + 1]
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ConversionError("No contours found in the mask.")

    coord_scale = 2 ** (p_scale - 1)
    hierarchy_row = hierarchy[0] if hierarchy is not None else []
    entries = []

    for index, contour in enumerate(contours):
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area:
            continue

        approx = cv2.approxPolyDP(contour, max(0.0, simplify), True)
        if len(approx) < 3:
            continue

        is_hole = bool(len(hierarchy_row) > index and hierarchy_row[index][3] >= 0)
        if is_hole:
            approx = approx[::-1]

        draw = contour_to_ass(approx, coord_scale)
        if draw:
            entries.append((area, draw, len(approx)))

    if not entries:
        raise ConversionError("All contours were filtered out.")

    entries.sort(key=lambda item: item[0], reverse=True)
    shape = " ".join(item[1] for item in entries)
    x0, y0, _, _ = bbox
    return [make_ass_text(shape, p_scale, pos_x + x0, pos_y + y0, blur)], {
        "engine": "opencv",
        "contours": len(entries),
        "points": sum(item[2] for item in entries),
        "chars": len(shape),
    }


def convert_with_vtracer(
    image: Image.Image,
    temp_dir: Path,
    args: argparse.Namespace,
) -> Path:
    try:
        import vtracer
    except Exception as exc:
        raise ConversionError(f"vtracer is not available: {exc}") from exc

    png_path = temp_dir / "trace_input.png"
    svg_path = temp_dir / "trace_output.svg"
    image.save(png_path)

    try:
        vtracer.convert_image_to_svg_py(
            str(png_path),
            str(svg_path),
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=args.filter_speckle,
            color_precision=args.color_precision,
            layer_difference=args.layer_difference,
            corner_threshold=args.corner_threshold,
            length_threshold=args.length_threshold,
            max_iterations=args.max_iterations,
            splice_threshold=args.splice_threshold,
            path_precision=args.path_precision,
        )
    except Exception as exc:
        raise ConversionError(f"vtracer failed: {exc}") from exc

    if not svg_path.exists() or svg_path.stat().st_size == 0:
        raise ConversionError("vtracer did not create an SVG.")
    return svg_path


def svg_to_ass(svg_path: Path, p_scale: int) -> list[str]:
    try:
        from defusedxml import ElementTree
        from svg2ssa.document import SVG
    except Exception as exc:
        raise ConversionError(f"svg2ssa is not available: {exc}") from exc

    doc = SVG()
    try:
        doc.from_svg_file(str(svg_path), ElementTree)
        ssa = doc.ssa_repr(
            {
                **doc.ssa_repr_config,
                "magnification_level": p_scale,
                "stroke_preservation": 0,
                "unnecessary_transformations": set(),
            }
        )
    except Exception as exc:
        raise ConversionError(f"svg2ssa failed: {exc}") from exc

    lines = []
    for line in ssa.splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            if len(parts) == 10:
                lines.append(parts[9].strip())
    if not lines:
        raise ConversionError("svg2ssa produced no dialogue lines.")
    return lines


def clean_svg2ssa_text(
    text: str,
    p_scale: int,
    pos_x: float,
    pos_y: float,
    offset_x: float,
    offset_y: float,
    blur: float,
    keep_color: bool,
) -> str:
    match = re.match(r"^\{([^}]*)\}\s*(.*?)\s*\{\\p0\}\s*$", text.strip())
    if not match:
        raise ConversionError(f"Unsupported svg2ssa dialogue text: {text[:120]}")

    tags, drawing = match.group(1), match.group(2).strip()
    local_x = 0.0
    local_y = 0.0
    pos_match = re.search(r"\\pos\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", tags)
    if pos_match:
        local_x = float(pos_match.group(1))
        local_y = float(pos_match.group(2))
    tags = re.sub(r"\\p\d+", "", tags)
    tags = re.sub(r"\\pos\([^)]*\)", "", tags)
    tags = re.sub(r"\\move\([^)]*\)", "", tags)
    tags = re.sub(r"\\org\([^)]*\)", "", tags)
    tags = re.sub(r"\\an\d+", "", tags)
    if not keep_color:
        tags = re.sub(r"\\(?:\d?c|[1234]c)&H[0-9A-Fa-f]+&", "", tags)
        tags = re.sub(r"\\(?:alpha|[1234]a)&H[0-9A-Fa-f]+&", "", tags)

    pos_x_i = int(round(pos_x + offset_x + local_x))
    pos_y_i = int(round(pos_y + offset_y + local_y))
    blur_tag = "" if blur <= 0 else f"\\blur{blur:g}"
    return f"{{\\an7\\pos({pos_x_i},{pos_y_i})\\bord0\\shad0{blur_tag}\\p{p_scale}{tags}}}{drawing}{{\\p0}}"


def vtracer_to_lines(
    trace_image: Image.Image,
    bbox: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> tuple[list[str], dict]:
    with tempfile.TemporaryDirectory(prefix="png2ass_") as temp_name:
        temp_dir = Path(temp_name)
        svg_path = convert_with_vtracer(trace_image, temp_dir, args)
        raw_lines = svg_to_ass(svg_path, args.p_scale)

    lines = [
        clean_svg2ssa_text(
            line,
            p_scale=args.p_scale,
            pos_x=args.pos_x,
            pos_y=args.pos_y,
            offset_x=bbox[0],
            offset_y=bbox[1],
            blur=args.blur,
            keep_color=args.keep_color,
        )
        for line in raw_lines
    ]
    chars = sum(len(line) for line in lines)
    return lines, {
        "engine": "vtracer+svg2ssa",
        "contours": len(lines),
        "points": None,
        "chars": chars,
    }


def write_text(path: str | None, text: str) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def read_input_list(path: str | None) -> list[str]:
    if not path:
        raise ConversionError("--input-list requires a file path.")
    try:
        content = Path(path).read_text(encoding="utf-8-sig")
    except Exception as exc:
        raise ConversionError(f"Could not read input list: {exc}") from exc

    inputs = [line.strip() for line in content.splitlines() if line.strip()]
    if not inputs:
        raise ConversionError("Input list is empty.")
    return inputs


def write_sequence(path: str | None, results: list[dict]) -> None:
    if not path:
        raise ConversionError("--sequence-out or --out is required with --input-list.")

    lines = ["PNG2ASS_SEQUENCE 1"]
    for index, result in enumerate(results, 1):
        ass_lines = result["ass_lines"]
        lines.append(f"FRAME {index}")
        lines.append(f"LINES {len(ass_lines)}")
        lines.extend(ass_lines)
    write_text(path, "\n".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PNG masks to ASS draw text.")
    parser.add_argument("--input", help="Input PNG path.")
    parser.add_argument("--input-list", help="UTF-8 text file with one PNG path per line.")
    parser.add_argument("--mode", choices=("auto", "alpha", "luma", "dark-matte", "white-matte", "color"), default="auto")
    parser.add_argument("--engine", choices=("auto", "vtracer", "opencv"), default="auto")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--p-scale", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--min-area", type=float, default=2.0)
    parser.add_argument("--max-chars", type=int, default=200000)
    parser.add_argument("--max-lines", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=4000000)
    parser.add_argument("--pos-x", type=float, default=0.0)
    parser.add_argument("--pos-y", type=float, default=0.0)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--keep-color", action="store_true")
    parser.add_argument("--filter-speckle", type=int, default=2)
    parser.add_argument("--color-precision", type=int, default=6)
    parser.add_argument("--layer-difference", type=int, default=16)
    parser.add_argument("--corner-threshold", type=int, default=60)
    parser.add_argument("--length-threshold", type=float, default=4.0)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--splice-threshold", type=int, default=45)
    parser.add_argument("--path-precision", type=int, default=3)
    parser.add_argument("--out", help="Write ASS override text lines to this file.")
    parser.add_argument("--sequence-out", help="Write grouped ASS override text for --input-list.")
    parser.add_argument("--log", help="Write plain error text to this file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON stdout/stderr.")
    parser.add_argument("--check-dependencies", action="store_true", help="Verify required Python modules.")
    parser.add_argument("--allow-many-lines", action="store_true", help="Allow output above max-lines.")
    parser.add_argument("--version", action="store_true", help="Print package version.")
    return parser.parse_args(argv)


def check_dependencies() -> dict:
    modules = {
        "Pillow": "PIL",
        "opencv-python": "cv2",
        "numpy": "numpy",
        "svg2ssa": "svg2ssa",
        "defusedxml": "defusedxml",
        "vtracer": "vtracer",
    }
    versions = {}
    for package, module_name in modules.items():
        module = importlib.import_module(module_name)
        versions[package] = getattr(module, "__version__", "available")
    return versions


def run(args: argparse.Namespace, input_path: str | None = None, write_output: bool = True) -> dict:
    source = input_path or args.input
    rgba, width, height = load_rgba(Path(source), args.max_pixels)
    mask, warnings, mode = build_mask(rgba, args.mode, args.threshold)
    bbox, trim_warnings = trim_mask(mask)
    warnings.extend(trim_warnings)
    trace_image = crop_for_trace(rgba, mask, bbox, mode)

    engine_errors = []
    lines = None
    stats = None

    if args.engine in {"auto", "vtracer"}:
        try:
            lines, stats = vtracer_to_lines(trace_image, bbox, args)
        except ConversionError as exc:
            if args.engine == "vtracer":
                raise
            engine_errors.append(str(exc))
            warnings.append(f"vtracer fallback: {exc}")

    if lines is None:
        lines, stats = mask_to_opencv_lines(
            mask,
            bbox,
            p_scale=args.p_scale,
            simplify=args.simplify,
            min_area=args.min_area,
            pos_x=args.pos_x,
            pos_y=args.pos_y,
            blur=args.blur,
        )

    if len(lines) > args.max_lines:
        message = f"Too many ASS lines ({len(lines)}). Raise max-lines or simplify the image."
        if not args.allow_many_lines:
            raise ConversionError(message)
        warnings.append(message)

    chars = sum(len(line) for line in lines)
    if chars > args.max_chars:
        raise ConversionError(
            f"ASS output is too large ({chars} chars). Increase simplify/filter-speckle or raise max-chars."
        )

    if write_output:
        write_text(args.out, "\n".join(lines))

    x0, y0, x1, y1 = bbox
    return {
        "ok": True,
        "mode": mode,
        "engine": stats["engine"],
        "width": width,
        "height": height,
        "bbox": [x0, y0, x1, y1],
        "draw_width": x1 - x0 + 1,
        "draw_height": y1 - y0 + 1,
        "p_scale": args.p_scale,
        "lines": len(lines),
        "contours": stats["contours"],
        "points": stats["points"],
        "chars": chars,
        "warnings": warnings,
        "engine_errors": engine_errors,
        "ass_text": lines[0] if len(lines) == 1 else None,
        "ass_lines": lines,
    }


def run_sequence(args: argparse.Namespace) -> dict:
    inputs = read_input_list(args.input_list)
    results = []
    warnings = []
    total_lines = 0
    total_chars = 0

    for index, input_path in enumerate(inputs, 1):
        try:
            result = run(args, input_path=input_path, write_output=False)
        except ConversionError as exc:
            name = Path(input_path).name
            raise ConversionError(f"Frame {index} ({name}): {exc}") from exc
        result["input"] = input_path
        results.append(result)
        total_lines += int(result["lines"])
        total_chars += int(result["chars"])
        warnings.extend(f"Frame {index}: {warning}" for warning in result["warnings"])

    write_sequence(args.sequence_out or args.out, results)
    return {
        "ok": True,
        "frames": len(results),
        "lines": total_lines,
        "chars": total_chars,
        "warnings": warnings,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.version:
        if not args.quiet:
            print(__version__)
        return 0
    if args.check_dependencies:
        try:
            versions = check_dependencies()
        except Exception as exc:
            write_text(args.log, str(exc))
            if not args.quiet:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
        if not args.quiet:
            print(json.dumps({"ok": True, "dependencies": versions}, ensure_ascii=False))
        return 0
    if args.input_list:
        try:
            result = run_sequence(args)
        except ConversionError as exc:
            write_text(args.log, str(exc))
            if not args.quiet:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
        except Exception as exc:
            message = f"Unexpected conversion error: {exc}"
            write_text(args.log, message)
            if not args.quiet:
                print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
            return 1

        write_text(args.log, "")
        if not args.quiet:
            print(json.dumps(result, ensure_ascii=False))
        return 0

    if not args.input:
        message = "--input or --input-list is required unless --check-dependencies or --version is used."
        write_text(args.log, message)
        if not args.quiet:
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1
    try:
        result = run(args)
    except ConversionError as exc:
        write_text(args.log, str(exc))
        if not args.quiet:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        message = f"Unexpected conversion error: {exc}"
        write_text(args.log, message)
        if not args.quiet:
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1

    write_text(args.log, "")
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False))
    return 0


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
