import sys

from pcdc_imvc.cli import train


def main():
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "multi"])
    train.main()


if __name__ == "__main__":
    main()