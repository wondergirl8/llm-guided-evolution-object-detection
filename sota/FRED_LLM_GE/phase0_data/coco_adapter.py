"""
Converts FRED annotations to YOLO format.

FRED format:  time: x1, y1, x2, y2, id, class  (corner format, pixels)
YOLO format:  class_id x_center y_center width height  (normalized 0-1)
"""

IMG_WIDTH = 1280
IMG_HEIGHT = 720


def parse_fred_annotation(line: str) -> dict | None:
    """
    Parse a single FRED annotation line.
    Returns dict with keys: time, x1, y1, x2, y2, track_id, drone_class
    Returns None if line is empty or malformed.
    """
    line = line.strip()
    if not line:
        return None

    try:
        time_part, rest = line.split(': ')
        parts = rest.split(', ')
        x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        track_id = int(parts[4])
        drone_class = parts[5].strip()

        return {
            'time': float(time_part),
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'track_id': track_id,
            'drone_class': drone_class,
        }
    except Exception as e:
        print(f"Warning: could not parse line '{line}': {e}")
        return None


def fred_to_yolo(annotation: dict, img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT) -> str:
    """
    Convert a FRED annotation dict to a YOLO format string.

    FRED: x1, y1, x2, y2 (pixel corners)
    YOLO: class_id x_center y_center width height (normalized 0-1)

    All drone types map to class 0 (single class: drone).
    """
    x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']

    # Convert corner format to center format
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width    = x2 - x1
    height   = y2 - y1

    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width    /= img_width
    height   /= img_height

    # Clamp to [0, 1] to handle any edge cases
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))

    # class_id is always 0 — single class (drone)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_annotation_file(annotation_path: str, output_path: str) -> int:
    """
    Convert a full FRED annotation file to YOLO format.
    Writes one YOLO label per line.
    Returns number of annotations converted.
    """
    annotations = []

    with open(annotation_path, 'r') as f:
        for line in f:
            ann = parse_fred_annotation(line)
            if ann:
                annotations.append(fred_to_yolo(ann))

    with open(output_path, 'w') as f:
        f.write('\n'.join(annotations))

    return len(annotations)


if __name__ == "__main__":
    # Quick test on a single annotation line
    test_line = "1.33332: 490.0, 413.0, 539.0, 448.0, 1, DJI Mini 2"
    ann = parse_fred_annotation(test_line)
    print("Parsed annotation:", ann)

    yolo = fred_to_yolo(ann)
    print("YOLO format:", yolo)

    # Verify the conversion makes sense
    parts = yolo.split()
    print(f"\nClass: {parts[0]}")
    print(f"x_center: {parts[1]} (should be ~{(490+539)/2/1280:.4f})")
    print(f"y_center: {parts[2]} (should be ~{(413+448)/2/720:.4f})")
    print(f"width:    {parts[3]} (should be ~{(539-490)/1280:.4f})")
    print(f"height:   {parts[4]} (should be ~{(448-413)/720:.4f})")
