"""Convenience entry point: python main.py --help"""
import argparse
from importlib import import_module
import sys


def main():
    parser = argparse.ArgumentParser(description="Educational chest X-ray image classification demo")
    parser.add_argument(
        "command",
        choices=["inspect", "train", "evaluate", "predict"],
    )
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    command_modules = {
        "inspect": "src.inspect_data",
        "train": "src.train",
        "evaluate": "src.evaluate",
        "predict": "src.predict",
    }
    command = import_module(command_modules[args.command])
    command.main()


if __name__ == "__main__":
    main()
