from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import sys

# Specific versions/commits
JAX_VERSION = "0.7.2"  # Replace with your desired version
JAX_MD_COMMIT = "6bd17d29ce5f9fe35a5582a42a9973b1ecd0859f"  # Replace with your specific commit hash
JAX_MD_URL = f"jax-md @ git+https://github.com/jax-md/jax-md.git@{JAX_MD_COMMIT}"


# JAX-MD dependencies (from its pyproject.toml, excluding jax/jaxlib)
JAX_MD_DEPS = [
    "absl-py",
    "numpy",
    "flax",
    "jraph",
    "einops",
    "ml_collections",
    "e3nn-jax",
    "dm-haiku",
    "optax",
    "frozendict",
    "pymatgen",
]

def install_jax_md():
    """Install JAX-MD from GitHub without dependencies."""
    jax_md_url = f"git+https://github.com/jax-md/jax-md.git@{JAX_MD_COMMIT}"
    print(f"\nInstalling JAX-MD from commit {JAX_MD_COMMIT} (without deps)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--no-deps", jax_md_url
    ])

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_jax_md()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_jax_md()

setup(
    name="vivarium",
    version="0.2.0",
    license="MIT",
    packages=find_packages(),    
        
    # Base JAX installation (CPU-only)
    install_requires=[
        f"jax=={JAX_VERSION}",
        f"jaxlib=={JAX_VERSION}",
        # JAX_MD_URL,
        "protobuf==5.29.5",
        "grpcio==1.76.0",
        "grpcio-tools==1.71.2",
        "panel==1.8.3",
        "param==2.2.1",
        "hydra-core==1.3.2",
        "psutil"
        # Add other dependencies here
    ] + JAX_MD_DEPS,
    
    # Optional dependencies for CUDA support
    extras_require={
        "cuda11": [
            f"jax[cuda11]=={JAX_VERSION}",
        ],
        "cuda12": [
            f"jax[cuda12]=={JAX_VERSION}",
        ],
        "cuda13": [
            f"jax[cuda13]=={JAX_VERSION}",
        ],
        # You can add more variants as needed
    },
    
    # Custom commands to install JAX-MD after other dependencies
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    

    author="Clément Moulin-Frier",
    author_email="clement.moulinfrier@gmail.com",
    python_requires=">=3.10",
    description="Vivarium enables configuring and running large-scale multi-agents simulations using Jax, with real-time interactions.",
)




# import io
# import os
# import setuptools

# # https://packaging.python.org/guides/making-a-pypi-friendly-readme/
# this_directory = os.path.abspath(os.path.dirname(__file__))
# with io.open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()

# setuptools.setup(
#     name="vivarium",
#     version="0.1.0",
#     license="MIT",
#     author="Clément Moulin-Frier",
#     author_email="clement.moulinfrier@gmail.com",
#     packages=setuptools.find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3.10",
#         # 'License :: OSI Approved :: Apache Software License',
#         "Operating System :: MacOS",
#         "Operating System :: POSIX :: Linux",
#         "Topic :: Scientific/Engineering",
#         "Intended Audience :: Science/Research",
#         "Intended Audience :: Developers",
#     ],
# )
