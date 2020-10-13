import argparse
import json
import logging
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Sequence

from fake_dataset import generate_fake_dataset


logger = logging.getLogger('dataset_fs_benchmark')

class TestFS:
    def __init__(self, base_path: Path, mount_target: Path, total_size: int, file_from: Path):
        self.base_path = base_path
        self.mount_target = mount_target
        self.total_size = total_size
        self.file_from = file_from

    def _copy_files(self):
        shutil.copytree(self.file_from, self.mount_target, dirs_exist_ok=True)

    def setup(self):
        raise NotImplementedError()
    def teardown(self):
        raise NotImplementedError()

    @contextmanager
    def test(self):
        logger.info('Setup filesystem %s', self.name)
        self.setup()
        try:
            yield self
        finally:
            time.sleep(5.0)  # hopefully avoid "device is busy"
            logger.info('Teardown filesystem %s', self.name)
            self.teardown()

class Link(TestFS):
    name = 'link'

    def setup(self):
        self.mount_target.symlink_to(self.file_from.relative_to(self.mount_target.parent))
    def teardown(self):
        self.mount_target.unlink()


class Ext4(TestFS):
    name = 'ext4'

    @property
    def fs_image_path(self):
        return self.base_path / 'dataset.ext4'

    def setup(self):
        fs_size_mb = int(self.total_size * 1.2 / 1024**2)
        subprocess.run(['dd', 'if=/dev/zero', f'of={self.fs_image_path}', 'bs=1M', f'count={fs_size_mb}'], check=True)
        subprocess.run(['mkfs.ext4', str(self.fs_image_path)], check=True)
        self.mount_target.mkdir()
        subprocess.run(['sudo', 'mount', '-o', 'loop', str(self.fs_image_path), str(self.mount_target)], check=True)
        subprocess.run(['sudo', 'cp', '-r', f'{self.file_from}/.' , str(self.mount_target)], check=True)
        subprocess.run(['sudo', 'umount', str(self.mount_target)], check=True)
        subprocess.run(['sudo', 'mount', '-o', 'loop', str(self.fs_image_path), str(self.mount_target)], check=True)

    def teardown(self):
        subprocess.run(['sudo', 'umount', str(self.mount_target)], check=True)
        self.mount_target.rmdir()
        self.fs_image_path.unlink()


class SquashFs(TestFS):
    name = 'squashfs'

    @property
    def fs_image_path(self):
        return self.base_path / 'dataset.sfs'

    def _mksquashfs(self):
        raise NotImplementedError()

    def setup(self):
        self._mksquashfs()
        self.mount_target.mkdir()
        subprocess.run(['sudo', 'mount', '-o', 'loop', str(self.fs_image_path), str(self.mount_target)], check=True)

    def teardown(self):
        subprocess.run(['sudo', 'umount', str(self.mount_target)], check=True)
        self.mount_target.rmdir()
        self.fs_image_path.unlink()

class SquashFs_AllUncompressed(SquashFs):
    name = 'squashfs-all-uncompressed'
    def _mksquashfs(self):
        subprocess.run([
            'mksquashfs', str(self.file_from), str(self.fs_image_path),
            '-noD', '-noI', '-noF', '-noX', '-no-duplicates', '-no-sparse',
            '-mem', '4G', '-processors', '4'], check=True,
        )

class SquashFs_DataUncompressed(SquashFs):
    name = 'squashfs-data-uncompressed'
    def _mksquashfs(self):
        subprocess.run([
            'mksquashfs', str(self.file_from), str(self.fs_image_path),
            '-noD', '-no-duplicates', '-no-sparse',
            '-mem', '4G', '-processors', '4'], check=True,
        )

class Erofs(TestFS):
    name = 'erofs'

    @property
    def fs_image_path(self):
        return self.base_path / 'dataset.erofs'

    def setup(self):
        subprocess.run(['mkfs.erofs', str(self.fs_image_path), str(self.file_from)], check=True)
        self.mount_target.mkdir()
        subprocess.run(['sudo', 'mount', '-o', 'loop', '-t', 'erofs', str(self.fs_image_path), str(self.mount_target)], check=True)

    def teardown(self):
        subprocess.run(['sudo', 'umount', str(self.mount_target)], check=True)
        self.mount_target.rmdir()
        self.fs_image_path.unlink()

class Zip(TestFS):
    @property
    def zip_path(self):
        return self.base_path / 'dataset.zip'
    def _zip(self):
        subprocess.run(['zip', '-0', '-q', '-r', str(self.zip_path.absolute()), '.'], cwd=self.file_from, check=True)
    def teardown(self):
        subprocess.run(['fusermount', '-u', str(self.mount_target)], check=True)
        self.mount_target.rmdir()
        self.zip_path.unlink()

class FuseZip(Zip):
    name='fuse-zip'
    def setup(self):
        self._zip()
        self.mount_target.mkdir()
        subprocess.run(['fuse-zip', '-r', str(self.zip_path), str(self.mount_target)])

class ArchiveMount(Zip):
    name='archivemount'
    def setup(self):
        self._zip()
        self.mount_target.mkdir()
        subprocess.run(['archivemount', '-o', 'readonly', str(self.zip_path), str(self.mount_target)])

def benchmark_worker(path_queue: Queue, exit_time: float, result_queue: Queue):
    completed_count = 0
    while True:
        t = time.perf_counter()
        p = path_queue.get()
        if p is None:
            break
        if t < exit_time:
            with p.open('rb') as f:
                f.read()
            completed_count += 1

    logger.debug('Worker finished. completed %d files', completed_count)
    result_queue.put(completed_count)
    result_queue.close()

def benchmark(all_files: Sequence[Path], queue_depth: int, drop_caches=0):
    logger.info('Benchmarking. queue depth: %d', queue_depth)
    if drop_caches > 0:
        subprocess.run(['sudo', 'sh', '-c', f'echo {drop_caches} > /proc/sys/vm/drop_caches'])

    path_queue = Queue()
    random.shuffle(all_files)

    end_t = time.perf_counter() + 30.0
    workers: List[Process] = []
    result_queue = Queue()
    for i in range(queue_depth):
        worker = Process(target=benchmark_worker, args=(path_queue, end_t, result_queue))
        worker.start()
        workers.append(worker)
    logger.debug('All workers started')

    start_t = time.perf_counter()
    for f in all_files:
        path_queue.put(f)
    for i in range(queue_depth):
        path_queue.put(None)
    path_queue.close()
    path_queue.join_thread()

    completed_count = 0
    for i in range(queue_depth):
        completed_count += result_queue.get()
    used_t = time.perf_counter() - start_t
    for w in workers:
        w.join()

    logger.info('Benchmark finished. read %d files in %f seconds', completed_count, used_t)
    return {'fileCount': completed_count, 'usedTime': used_t}


def size(size_str):
    unit = size_str[-1].lower()
    units = 'kmgtp'
    for i, u in enumerate(units):
        if u == unit:
            return int(round(float(size_str[:-1]) * (1024 ** (i + 1))))
    return int(size_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, default=Path('dataset_fs_benchmark'))
    parser.add_argument('--file-size', type=size, default=32 * 1024)
    parser.add_argument('--total-size', type=size, default=2 * (1024 ** 3))
    parser.add_argument('-o', '--output', type=Path, default=Path('results.json'))
    args = parser.parse_args()

    if args.path.exists():
        raise RuntimeError(f'Path "{args.path}" already exists.')

    logger.info('Setup tmpfs at %s', args.path)
    args.path.mkdir()
    subprocess.run(['sudo', 'mount', '-t', 'tmpfs', 'tmpfs', str(args.path)], check=True)

    dataset_path = args.path / 'dataset'
    all_files = generate_fake_dataset(
        dir=dataset_path,
        file_size=args.file_size,
        total_size=args.total_size,
    )

    mount_target = args.path / 'dataset_fs'
    all_read_files = [mount_target / p.relative_to(dataset_path) for p in all_files]

    all_fs = [Link, Ext4, SquashFs_AllUncompressed, SquashFs_DataUncompressed, Erofs, FuseZip, ArchiveMount]
    results = {}
    if args.output.exists():
        with args.output.open('r') as f:
            results = json.load(f)
    assert isinstance(results, dict)

    for fs_class in all_fs:
        fs: TestFS = fs_class(
            base_path=args.path,
            mount_target=mount_target,
            total_size=args.total_size,
            file_from=dataset_path,
        )
        fs_results = []
        with fs.test():
            logger.info('Warmup')
            benchmark(all_read_files, queue_depth=4)
            logger.info('Benchmarking fs: %s', fs.name)
            for qdepth in [1, 8, 64]:
                res = benchmark(all_read_files, queue_depth=qdepth, drop_caches=3)
                res['queueDepth'] = qdepth
                res['dropCaches'] = 3
                fs_results.append(res)
            results[fs.name] = fs_results
            with args.output.open('w') as f:
                json.dump(results, f, indent=4)

    logger.info('Cleanup')
    subprocess.run(['sudo', 'umount', str(args.path)], check=True)
    args.path.rmdir()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
