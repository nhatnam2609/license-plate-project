from pathlib import Path

from . import create_training_list 
from . import create_dict_file
from . import create_config


def prepare_training():
    """
    Prepare all necessary files for PaddleOCR training
    """
    # Create output directory
    output_path = Path('paddleocr_training')
    output_path.mkdir(exist_ok=True)

    # Create training lists
    for split in ['train', 'val', 'test']:
        create_training_list('dataset', output_path, split)

    # Create dictionary
    num_classes = create_dict_file('dataset', output_path)

    # Create configuration
    create_config(output_path, num_classes)

    print("\nTraining preparation completed!")