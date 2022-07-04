from setuptools import setup

setup(name="gym_hpc",
      version="1.0",
      url="https://github.com/hpc-unex/gym-hpc",
      author="Juan A. Rico-Gallego (jarico@unex.es)",
      license="BSD",
      # packages=["gym_hpc", "gym_hpc.envs"],
      install_requires = ["gym", "torch", "numpy"]
)
