from setuptools import setup, find_packages

# TODO: clean commented lines at some point.
# Haven't done it yet in case something messes in the compatibility between jax-md and the last version of jax
# In that case, we might need to first install a specific version of jax,
# then install jax-md from without its dependencies, and finally install jax-md dependencies manually

# Specific versions/commits
# JAX_VERSION = "0.7.2"  # Replace with your desired version
JAX_MD_COMMIT = "6bd17d29ce5f9fe35a5582a42a9973b1ecd0859f"  # Replace with your specific commit hash
JAX_MD_URL = f"jax-md @ git+https://github.com/jax-md/jax-md.git@{JAX_MD_COMMIT}"


# JAX-MD dependencies (from its pyproject.toml, excluding jax/jaxlib) 
# JAX_MD_DEPS = [
#     "absl-py",
#     "numpy",
#     "flax",
#     "jraph",
#     "einops",
#     "ml_collections",
#     "e3nn-jax",
#     "dm-haiku",
#     "optax",
#     "frozendict",
#     "pymatgen",
# ]


setup(
    name="vivarium",
    version="0.2.0",
    license="MIT",
    packages=find_packages(),    
        
    # Base JAX installation (CPU-only)
    install_requires=[
        "jax",
        "jaxlib",
        # f"jax=={JAX_VERSION}",
        # f"jaxlib=={JAX_VERSION}",
        JAX_MD_URL,
        "protobuf==5.29.5",
        "grpcio==1.76.0",
        "grpcio-tools==1.71.2",
        "panel==1.8.3",
        "param==2.2.1",
        "hydra-core==1.3.2",
        "psutil",
        "pytest"
        # Add other dependencies here
    ], # + JAX_MD_DEPS,
    
    # Optional dependencies for CUDA support
    extras_require={
        "cuda11": [
            f"jax[cuda11]", #=={JAX_VERSION}",
        ],
        "cuda12": [
            f"jax[cuda12]", #=={JAX_VERSION}",
        ],
        "cuda13": [
            f"jax[cuda13]", #=={JAX_VERSION}",
        ],
        "ngrok": [
            "pyngrok",
        ],
        # You can add more variants as needed
    },
    
    author="Clément Moulin-Frier",
    author_email="clement.moulinfrier@gmail.com",
    python_requires=">=3.10",
    description="Vivarium enables configuring and running large-scale multi-agents simulations using Jax, with real-time interactions.",
)
