#!/usr/bin/env python3
"""
Compatibility wrapper. Use convert_data.py instead.
"""

from __future__ import annotations

from convert_data import Args, main, tyro


if __name__ == "__main__":
    main(tyro.cli(Args))
