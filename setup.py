from setuptools import setup, find_packages

setup(
    name="vivarium",
    version="0.2.0",
    license="MIT",
    packages=find_packages(),    
        
    install_requires=[
        # JAX 0.8.x only supports Apple Silicon and Linux/Windows
        # Intel Macs (x86_64 darwin) need older JAX 0.4.x and jax-md from specific commit
        "jax==0.8.2; platform_machine != 'x86_64' or sys_platform != 'darwin'",
        "jaxlib==0.8.2; platform_machine != 'x86_64' or sys_platform != 'darwin'",
        "jax-md==0.2.27; platform_machine != 'x86_64' or sys_platform != 'darwin'",
        "jax==0.4.38; platform_machine == 'x86_64' and sys_platform == 'darwin'",
        "jaxlib==0.4.38; platform_machine == 'x86_64' and sys_platform == 'darwin'",
        "jax-md @ git+https://github.com/jax-md/jax-md.git@6bd17d29ce5f9fe35a5582a42a9973b1ecd0859f; platform_machine == 'x86_64' and sys_platform == 'darwin'",
        "protobuf==5.29.5",
        "grpcio==1.71.2",
        "grpcio-tools==1.71.2",
        "grpcio-health-checking==1.71.2",
        "panel==1.8.5",
        "param==2.3.1",
        "hydra-core==1.3.2",
        "psutil==7.2.1",
        "pytest==9.0.2",
        "pytest-timeout>=2.3.1",
        "python-dotenv==1.2.1",
        "notebook==7.0.8",
    ],
    
    extras_require={
        "cuda11": ["jax[cuda11]"],
        "cuda12": ["jax[cuda12]"],
        "cuda13": ["jax[cuda13]"],
        "ngrok": ["pyngrok"],
    },
    
    author="Clément Moulin-Frier",
    author_email="clement.moulinfrier@gmail.com",
    python_requires=">=3.11",
    description="Vivarium enables configuring and running large-scale multi-agents simulations using Jax, with real-time interactions.",
)
