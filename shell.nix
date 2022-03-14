let
  jupyter = import (builtins.fetchGit {
    url = https://github.com/tweag/jupyterWith;
    # Example working revision, check out the latest one.
    rev = "45f9a774e981d3a3fb6a1e1269e33b4624f9740e";
  }) {};

  nixpkgs = import (builtins.fetchGit {
    url = github:NixOS/nixpkgs/nixos-21.05;
  }) {};

  NanoParticleTools = systemString:
    with import nixpkgs;
    with python38Packages;
    buildPythonPackage {
      pname = "NanoParticleTools";
      version = "0.1";
      src = ./.;
      checkInputs = [
        pymatgen
        monty
        (builtins.getAttr systemString RNMC.defaultPackage)
        sqlite
      ];

    };


  iPython = jupyter.kernels.iPythonWith {
    name = "python";
    packages = p: with p; [ numpy NanoParticleTools];
  };

  jupyterEnvironment =
    jupyter.jupyterlabWith {
      kernels = [ iPython ];
    };
in
  jupyterEnvironment.env