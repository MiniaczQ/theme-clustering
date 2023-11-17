# Based on https://gist.github.com/JosephCatrambone/baaef25d338dd6b8b332e76a0445ba0d

import itertools
import sys

import numpy as np
import torch
from scipy.cluster.vq import kmeans, kmeans2, whiten
from transformers import AutoTokenizer, GPT2Model
from typing import List
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def magnitude(x):
    return np.dot(x, x) ** 0.5


def distance(a, b):
    delta = b - a
    return np.dot(delta, delta) ** 0.5


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")


def embeddings_from_strings(lines: List[str], normalize: bool = True) -> NDArray:
    all_embeddings = list()
    for line in lines:
        tokens = tokenizer(line, return_tensors="pt")
        model_output = model(**tokens)
        embeddings = model_output.last_hidden_state.detach().numpy()[0, 0, :]
        if normalize:
            embeddings /= magnitude(embeddings)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


def reduce_dimensions(embeddings: NDArray, n_dims: int):
    reducer = PCA(n_components=n_dims)
    reduced = reducer.fit_transform(embeddings)
    return reduced


def main(filename, num_clusters):
    num_clusters = int(num_clusters)
    with open(filename, "rt") as fin:
        lines = list(line.strip() for line in fin.readlines())

    embeddings = embeddings_from_strings(lines)
    reduced = reduce_dimensions(embeddings, num_clusters)
    (min, max) = np.min(reduced, axis=0), np.max(reduced, axis=0)
    reduced /= min
    reduced -= max - min

    plt.scatter(reduced[:, 0], reduced[:, 1])
    for ([x, y], line) in zip(reduced, lines):
        plt.text(x, y, line)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
