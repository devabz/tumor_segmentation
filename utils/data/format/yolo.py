

def create_yaml_config(path: str, train: str, val: str, names: dict[int, str], test: str = None) -> dict:
    """
    Create a YAML configuration for YOLO training.

    :param path: Path to save the YAML file.
    :param train: Path to training images.
    :param val: Path to validation images.
    :param names: Dictionary of class names.
    :param test: Optional path to test images.
    :return: Dictionary containing the YAML configuration.
    """

    config = {
        'path': path,
        'train': train,
        'val': val,
        'names': names,
    }
    
    if test:
        config['test'] = test

    return config


def create_label(class_id: int, bounding_coordinates: tuple, img_width: int, img_height: int, decimals=3) -> str:
    """
    Create a YOLO formatted label string.

    :param class_id: Class ID for the object.
    :param bounding_coordinates: Tuple containing (x_center, y_center, width, height).
    :return: Formatted label string.
        """
    
    def format_coordinates(coords):
        def normalize(coord):
            x, y = coord
            x, y = x/img_width, y/img_height
            return ' '.join(map(str, (round(x, decimals), round(y, decimals))))
        
        coords = list(map(normalize, coords))
        coords = ' '.join(coords)

        return f"{class_id} {coords}"
    
    return '\n'.join(map(
        format_coordinates, 
        filter(
            lambda x: len(x) >= 3, 
            bounding_coordinates
        )
    ))