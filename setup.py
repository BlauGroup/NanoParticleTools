from setuptools import setup, find_packages

setup(name='NanoParticleTools',
      version='0.1',
      description='NanoParticleTools',
      url='https://github.com/BlauGroup/NanoParticleTools',
      author='Daniel Barter',
      author_email='danielbarter@gmail.com',
      packages=find_packages("src"),
      package_dir={"": "src"},
      install_requires=[
          "setuptools",
          "numpy",
          "pymatgen",
          "fireworks",
          "monty",
          "jobflow",
          "atomate2",
          "maggma",
          "pytest",
          "torch==1.12.1",
          "pytorch-lightning==1.7.7",
          "ray",
          "wandb",
          "gpuparallel",
          "torch-geometric<=2.3.1",
          "h5py",
          "torch-scatter==2.1.0"  
      ]
      )
