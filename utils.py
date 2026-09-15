from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image


def load_target_images_from_directory(
    directory_path: Union[str, Path],
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    recursive: bool = False
) -> List[TargetImage]:
    """
    Scans a directory for image files and instantiates a TargetImage object for each.

    Args:
        directory_path: Path to the target directory containing images.
        extensions: Tuple of file extensions (case-insensitive) to include.
        recursive: If True, recursively searches subdirectories.

    Returns:
        A list of initialized TargetImage instances.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    pattern = "**/*" if recursive else "*"
    valid_exts = {ext.lower() for ext in extensions}
    
    image_files = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    target_images = []
    for img_path in sorted(image_files):
        try:
            # Load RGBA image via PIL and convert to NumPy array
            pil_img = Image.open(img_path).convert("RGBA")
            rgba_arr = np.array(pil_img)

            # Instantiate TargetImage directly via dataclass constructor
            target_img = TargetImage(
                name=img_path.stem,
                image_path=img_path,
                rgba_image=rgba_arr
            )
            target_images.append(target_img)
        except Exception as e:
            print(f"Warning: Failed to load '{img_path.name}': {e}")

    return target_images


def extract_segment_count_from_filename(
    image_path: Union[str, Path], 
    default: Optional[int] = None
) -> Optional[int]:
    """
    Extracts the segment count encoded at the end of an image filename.

    Examples:
        "scene_04.png"          -> 4
        "stimulus_seg_12.jpg"   -> 12
        "target_image3.png"     -> 3

    Args:
        image_path: Path or filename of the target image.
        default: Fallback integer to return if no trailing digits are found.

    Returns:
        The extracted segment count as an integer, or the default value.
    """
    stem = Path(image_path).stem  # Removes directory path and extension
    
    # Matches one or more digits at the end of the string
    match = re.search(r'(\d+)$', stem)
    
    if match:
        return int(match.group(1))
    
    return default
