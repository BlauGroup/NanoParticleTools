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
          "pytest"
      ]
      )
