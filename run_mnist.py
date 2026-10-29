"""Train the mnist example through the shared command-line interface."""

import sys

from dl_example.cli import main

if __name__ == "__main__":
    main(["train", "--dataset", "mnist", *sys.argv[1:]])
