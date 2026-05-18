IMAGE_SIZE_OPTIONS = ["auto", "1024x1024", "1024x1536", "1536x1024"]
MINIMUM_IMAGE_SIZE = "1024x1024"


def normalize_image_size(size: str) -> str:
    if size in IMAGE_SIZE_OPTIONS:
        return size
    return MINIMUM_IMAGE_SIZE
