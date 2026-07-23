#!/usr/bin/env python3
"""Train Markush recognition model."""


def main():
    from glyph.markush.train.sft import main as sft_main

    sft_main()


if __name__ == "__main__":
    main()
