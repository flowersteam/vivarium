from setuptools import setup, find_packages

setup(
    name="vivarium",
    version="0.2.0",
    license="MIT",
    packages=find_packages(),    
        
    install_requires=[
        "jax==0.8.2",
        "jaxlib==0.8.2",
        "jax-md==0.2.27",
        "protobuf==5.29.5",
        "grpcio==1.71.2",
        "grpcio-tools==1.71.2",
        "grpcio-health-checking==1.71.2",
        "panel==1.8.5",
        "param==2.3.1",
        "hydra-core==1.3.2",
        "psutil==7.2.1",
        "pytest==9.0.2",
        "python-dotenv==1.2.1",
    ],
    
    extras_require={
        "cuda11": ["jax[cuda11]"],
        "cuda12": ["jax[cuda12]"],
        "cuda13": ["jax[cuda13]"],
        "ngrok": ["pyngrok"],
    },
    
    author="Clément Moulin-Frier",
    author_email="clement.moulinfrier@gmail.com",
    python_requires=">=3.10",
    description="Vivarium enables configuring and running large-scale multi-agents simulations using Jax, with real-time interactions.",
)
