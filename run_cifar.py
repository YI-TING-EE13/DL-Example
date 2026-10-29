"""Train the cifar10 example through the shared command-line interface."""

import sys

from dl_example.cli import main

if __name__ == "__main__":
    main(["train", "--dataset", "cifar10", *sys.argv[1:]])
