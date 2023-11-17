from pathlib import Path
from typing import List


def preprocess_data(source: Path, dest: Path, recreate: bool = False):
    if dest.exists() and not recreate:
        return

    with open(source, "r", encoding="utf-8") as fin:
        with open(dest, "w") as fout:
            for i, line in enumerate(fin):
                if i % 2 != 0:
                    semicolon = line.find(";")
                    separate = 0 if semicolon == -1 else semicolon + 2
                    fout.write(line[separate:])


def load_data(source: Path) -> List[str]:
    with open(source, "rt") as fin:
        return list(line.strip() for line in fin.readlines())
