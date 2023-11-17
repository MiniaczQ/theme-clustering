from pathlib import Path
import sys


def main(source: Path, dest: Path):
    with open(source, "r", encoding="utf-8") as fin:
        with open(dest, "w") as fout:
            for i, line in enumerate(fin):
                if i % 2 != 0:
                    fout.write(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
