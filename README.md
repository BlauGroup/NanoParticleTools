NanoParticleTools tools is a python module that facilitates monte carlo simulation of Upconverting Nanoparticles (UCNP) using [RNMC](https://github.com/BlauGroup/RNMC).

After cloning the package, Install the package using the command:
```bash
python -m pip install -e .
```

[RNMC](https://github.com/BlauGroup/RNMC)must be installed to perform simulations, otherwise only input generation and analysis is available.


When running on a system with root access, a prebuilt NixOS environment may be used. The environment includes access to a compiled RNMC. To access the Nix development shell
```
nix develop
```

To launch a jupyter server use the command:
```bash
nix-shell --command "jupyter-lab"
```
