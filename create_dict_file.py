from pathlib import Path


def create_dict_file(dataset_path, output_path):
    """
    Create dictionary file containing all characters in the dataset
    """
    chars = set()

    # Collect all unique characters from all splits
    for split in ['train', 'val', 'test']:
        data_dir = Path(dataset_path) / split
        for img_path in data_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                chars.update(img_path.stem)

    # Sort and write to file
    chars = sorted(list(chars))
    dict_file = Path(output_path) / 'dict.txt'

    with open(dict_file, 'w', encoding='utf-8') as f:
        for char in chars:
            f.write(f'{char}\n')

    print(f"Created dictionary file at: {dict_file}")
    return len(chars)