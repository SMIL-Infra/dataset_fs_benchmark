from ast import parse
import json
import sys
from typing import Dict
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path, default=Path('results.json'), help='output of benchmark.py')
    parser.add_argument('--queue-depth', default=64)
    args = parser.parse_args()

    fig, ax = plt.subplots()
    with args.results.open('r') as f:
        results: Dict = json.load(f)

    labels = results.keys()
    qps = [next(r['fileCount'] / r['usedTime'] for r in res if r['queueDepth'] == args.queue_depth) for res in results.values()]
    qps, labels = zip(*sorted(zip(qps, labels), reverse=True))

    ax.bar(labels, qps)
    ax.set_xticklabels(labels, rotation=-30, ha='left', rotation_mode="anchor")
    ax.set_ylabel('files / s')
    fig.tight_layout()
    fig.savefig(f'plots/queueDepth_{args.queue_depth}.svg')

if __name__ == "__main__":
    main()
