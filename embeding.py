import numpy as np
from transformers import AutoTokenizer, GPT2Model
from typing import List
from numpy.typing import NDArray


def magnitude(x):
    return np.dot(x, x) ** 0.5


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")


def embed(lines: List[str], normalize: bool = True) -> NDArray:
    embeddings = list()
    for line in lines:
        tokens = tokenizer(line, return_tensors="pt")
        model_output = model(**tokens)
        embedding = model_output.last_hidden_state.detach().numpy()[0, 0, :]
        if normalize:
            embedding /= magnitude(embedding)
        embeddings.append(embedding)
    return np.vstack(embeddings)
