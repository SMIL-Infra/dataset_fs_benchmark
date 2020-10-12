import logging
from pathlib import Path
import os
import random
import string
from tqdm import tqdm


logger = logging.getLogger(__name__)

def new_path(in_dir: str):
    while True:
        p = random.choices(string.ascii_letters + string.digits, k=12)
        p = in_dir / (''.join(p) + '.dat')
        if not p.exists():
            return p

def generate_fake_dataset(dir: Path, file_size: int, total_size: int):
    file_count = total_size // file_size
    logger.info('Generating fake dataset files at %s. %d files of %d bytes', dir, file_size, file_count)

    assert not dir.exists()
    dir.mkdir()

    all_files = []
    for i in tqdm(range(file_count)):
        data = os.urandom(file_size)
        path = new_path(dir)
        all_files.append(path)
        with path.open('wb') as f:
            f.write(data)

    return all_files
