# kite-png2ass

Python package for the `PNG2ASS` Aegisub macro.

It converts PNG images into ASS drawing text and can be used either from the macro or directly from the command line.

Macro documentation: <https://github.com/Kitherow/Kite-Aegisub-Scripts/blob/main/docs/PNG2ASS.md>

## Modes

- `auto`: use alpha if present, otherwise detect binary mattes or color images.
- `alpha`: trace visible alpha.
- `white-matte`: trace white foreground over black background.
- `dark-matte`: trace dark foreground over light background.
- `luma`: alias for dark luminance after compositing transparency over white.
- `color`: trace the visible canvas and preserve color when `--keep-color` is used.

## Engines

- `auto`: try `vtracer` first and fall back to OpenCV contours.
- `vtracer`: use `vtracer` and `svg2ssa`.
- `opencv`: use OpenCV contours directly.

## Output

The output file contains one ASS text payload per line. The macro inserts those payloads as new dialogue lines with drawing mode enabled.

## License

MIT. See `LICENSE`.
