import sys
import matplotlib.pyplot as plt
from clustering import cluster_kmeans, normalized_pca
from pathlib import Path

from embeding import embed
from preprocess import load_data, preprocess_data


def show(data, labels):
    plt.scatter(data[:, 0], data[:, 1])
    for [x, y], line in zip(data, labels):
        plt.text(x, y, line)
    plt.show()


def main(raw: Path, processed: Path, n: int):
    n = int(n)
    preprocess_data(raw, processed, True)

    lines = load_data(processed)
    embeddings = embed(lines)
    clusters = cluster_kmeans(embeddings, lines, n)
    for idx, cluster in enumerate(clusters):
        entries = '\n'.join(["- " + e for e in cluster])
        print(f"Cluster {idx}:\n{entries}")


if __name__ == "__main__":
    main(Path("data/raw/data.txt"), Path("data/processed/data.txt"), 30)
