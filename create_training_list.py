import re
from pathlib import Path

def create_training_list(dataset_path, output_path, split='train'):
    """
    Create training list file in PaddleOCR format
    Format: image_path/label
    """
    data_dir = Path(dataset_path) / split
    output_file = Path(output_path) / f'{split}_list.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        for img_path in data_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Get label from filename
                label = img_path.stem
                label = re.sub(r'_.*', '', label) 

                rel_path = img_path.relative_to(Path(dataset_path))
                f.write(f'{rel_path}\t{label}\n')

    print(f"Created {split} list at: {output_file}")
