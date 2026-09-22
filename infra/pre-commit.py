#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump the hash to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@d54272cad0f0823ab9a00e57fddcc169e9fcf3d2",
# ]
# ///
"""Run shared checks for the Marin-owned Axolotl delta."""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
