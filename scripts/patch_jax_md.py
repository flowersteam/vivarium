#!/usr/bin/env python3
"""
Patch jax-md for compatibility with JAX 0.4.24+ where jax.random.KeyArray was removed.

This script replaces `KeyArray = random.KeyArray` with `KeyArray = jax.Array` in jax_md/rigid_body.py.
Run this after installing jax-md on platforms that require older JAX versions (e.g., Intel Mac).
"""

import sys
import importlib.util


def find_jax_md_path():
    """Find the installation path of jax_md."""
    spec = importlib.util.find_spec("jax_md")
    if spec is None or spec.origin is None:
        print("Error: jax_md is not installed")
        sys.exit(1)

    # spec.origin points to __init__.py, get the package directory
    import os
    return os.path.dirname(spec.origin)


def patch_rigid_body(jax_md_path):
    """Patch rigid_body.py to use jax.Array instead of random.KeyArray."""
    import os

    rigid_body_path = os.path.join(jax_md_path, "rigid_body.py")

    if not os.path.exists(rigid_body_path):
        print(f"Error: {rigid_body_path} not found")
        sys.exit(1)

    with open(rigid_body_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "KeyArray = jax.Array" in content:
        print("jax_md/rigid_body.py is already patched")
        return False

    # Check if patch is needed
    if "KeyArray = random.KeyArray" not in content:
        print("Warning: Expected pattern 'KeyArray = random.KeyArray' not found")
        print("The file may have a different structure or already be patched differently")
        return False

    # Apply patch
    new_content = content.replace(
        "KeyArray = random.KeyArray",
        "KeyArray = jax.Array  # Patched for JAX 0.4.24+ compatibility"
    )

    with open(rigid_body_path, "w") as f:
        f.write(new_content)

    print(f"Successfully patched {rigid_body_path}")
    return True


def main():
    print("Patching jax-md for JAX 0.4.24+ compatibility...")
    jax_md_path = find_jax_md_path()
    print(f"Found jax_md at: {jax_md_path}")
    patch_rigid_body(jax_md_path)
    print("Done!")


if __name__ == "__main__":
    main()
