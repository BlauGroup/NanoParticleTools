![logo](./logo_transparent.png)

[linting-image]: https://github.com/BlauGroup/NanoParticleTools/actions/workflows/flake8.yml/badge.svg
[linting-url]: https://github.com/BlauGroup/NanoParticleTools/actions/workflows/flake8.yml
[testing-image]: https://github.com/BlauGroup/NanoParticleTools/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/BlauGroup/NanoParticleTools/actions/workflows/testing.yml
[coverage-image]: https://codecov.io/gh/BlauGroup/NanoParticleTools/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/github/BlauGroup/NanoParticleTools?branch=main

[![Test Coverage][coverage-image]][coverage-url]
[![Linting Status][linting-image]][linting-url]
[![Testing Status][testing-image]][testing-url]
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)


NanoParticleTools tools is a python module that facilitates monte carlo simulation of Upconverting Nanoparticles (UCNP) using [RNMC](https://github.com/BlauGroup/RNMC) and analysis/prediction using deep learning, detailed in the [manuscript by Sivonxay et. al](https://chemrxiv.org/engage/chemrxiv/article-details/6769dc3a81d2151a02b75ef6). 

# Using NanoParticleTools
NanoParticleTools provides functionality to generate inputs for running Monte Carlo Simulations on nanoparticles and analyzing outputs. Monte Carlo simulation uses NMPC within the [RNMC](https://github.com/BlauGroup/RNMC) package. While NanoParticleTools provides wrapper functions to run the C++ based simulator, [RNMC](https://github.com/BlauGroup/RNMC) must be installed to perform simulations.

Using only the machine learning capabilities within NanoParticleTools does not require the installation of [RNMC](https://github.com/BlauGroup/RNMC)
## Installation
To install NanoParticleTools to a python environment, clone the repository and use one of the following commands from within the NanoParticleTools directory
```bash
python setup.py develop
```
or 
```bash
pip install .
```

Finally, install torch-scatter as follows.
```
git clone https://github.com/rusty1s/pytorch_scatter.git
pip install pytorch_scatter/
```
 Note: This package is installed separately due to some installation issues. See: https://github.com/rusty1s/pytorch_scatter/issues/265 and https://github.com/rusty1s/pytorch_scatter/issues/424


Installation should take around 15 minutes on a normal desktop computer. NanoParticleTools can run on Python 3.10 and greater. The [setup.py](https://github.com/BlauGroup/NanoParticleTools/blob/main/setup.py) file includes pinned/constrained dependencies necessary for the installation. 

### Training and Using Machine Learning Models

The functionality to train and use deep learning models to predict UCNP emission intensity using NanoParticleTools is embedded [here](https://github.com/BlauGroup/NanoParticleTools/tree/main/src/NanoParticleTools/machine_learning). 

Within the demos folder, we have included Jupyter notebooks with [demos](https://github.com/BlauGroup/NanoParticleTools/src/NanoParticleTools/machine_learning/demos/) for (1) training a deep learning model on pre-compiled datasets of [RNMC](https://github.com/BlauGroup/RNMC) trajectories, (2) loading pre-trained models and predicting emission intensity for an arbitrary UCNP design, and (3) loading pre-trained models and predicting emission intensity on the pre-compiled UCNP datasets. 

Pre-compiled UCNP [datasets](https://figshare.com/s/49222bae78f228363897) (SUNSET) and pre-trained [models](https://figshare.com/articles/dataset/Hetero-GNN_Checkpoints/27941694/1?file=50919813)  
 can be downloaded from Figshare.

## Running Simulations
An example of local execution can be seen below.

```python
from NanoParticleTools.flows.flows import get_npmc_flow
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint

constraints = [SphericalConstraint(20)]
dopant_specifications = [(0, 0.1, 'Yb', 'Y'),
                         (0, 0.02, 'Er', 'Y')]

npmc_args = {'npmc_command': <NPMC_command>,
             'num_sims':2,
             'base_seed': 1000,
             'thread_count': 8,
             'simulation_length': 1000,
             }
spectral_kinetics_args = {'excitation_power': 1e12,
                          'excitation_wavelength':980}

flow = get_npmc_flow(constraints = constraints,
                     dopant_specifications = dopant_specifications,
                     doping_seed = 0,
                     spectral_kinetics_args = spectral_kinetics_args,
                     npmc_args = npmc_args,
                     output_dir = './scratch')
```

```python
from jobflow import run_locally
from maggma.stores import MemoryStore
from jobflow import JobStore

# Store the output data locally in a MemoryStore
docs_store = MemoryStore()
data_store = MemoryStore()
store = JobStore(docs_store, additional_stores={'trajectories': data_store})

responses = run_locally(flow, store=store, ensure_success=True)
```

In this example, the target `maggma.stores.MemoryStore` used to collect output is volatile and will be lost if the Store is reinitialized or the python kernel is restarted. Therefore, one may opt to use a MongoDB server to save calculation output to ensure data persistence. To integrate a MongoDB, use a MongoStore instead of a MemoryStore.
```
from maggma.stores.mongolike import MongoStore
docs_store = MongoStore(<mongo credentials or URI here>)
data_store = MongoStore(<mongo credentials or URI here>)
```
Refer to the maggma [Stores documentation](https://materialsproject.github.io/maggma/getting_started/stores/) for more information.

## Running the Builder
After running simulations, you may wish to average the outputs of trajectories obtained from the same recipe (using different dopant and simulation seeds). We have included a maggma builder in NanoParticleTools to easily group documents and perform the averaging. More information on builders can be found in the maggma [Builder documentation](https://materialsproject.github.io/maggma/reference/core_builder/)

An example of instantiating a builder is as follows:
```
from maggma.stores.mongolike import MongoStore

source_store = MongoStore(collection_name = "docs_npmc", <mongo credentials here>)
target_store = MongoStore(collection_name = "avg_npmc", <mongo credentials here>)

builder = UCNPBuilder(source_store, target_store, docs_filter={'data.simulation_time': {'$gte': 0.01}}, chunk_size=4)
```
Here, the `source_store` is a maggma Store which contains the trajectory documents produced from the SimulationReplayer analysis of NPMC runs. `target_store` is the Store in which you would like your averaged documents to be populated to. Optional arguments include a `docs_filter`, which is a pymongo query to target specific documents. `chunk_size` may also be specified and is dependent on the memory and speed of the machine executing the builder.

To execute the builder locally, use the `builder.run()` function. The builder may also be run in parallel or distributed mode, see the maggma ["Running a Builder Pipeline" documentation](https://materialsproject.github.io/maggma/getting_started/running_builders/)


# Contributing 
If you wish to make changes to NanoParticle tools, it may be wise to install the package in development mode. After cloning the package, use the following command.
```bash
python -m pip install -e .
```
Modifications should now be reflected when you run any functions in NanoParticleTools.

```
pytest --cov-report term-missing --cov=src tests/
```
Further guidance on contributing via Pull Requests will be added in the near future.
