{
  description = "Multi node reaction network generator";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-21.05;
    RNMC.url = github:BlauGroup/RNMC;
  };

  outputs = { self, nixpkgs, RNMC }:

    let

      NanoParticleTools = systemString:
        with import nixpkgs { system = systemString; };
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


      genericDevShell = systemString: installNanoParticleTools:
        with import nixpkgs { system = systemString; };
        mkShell {
          buildInputs = with python38Packages; [
            pymatgen
            monty
            numpy
            matplotlib
            (if installNanoParticleTools then (NanoParticleTools systemString) else null)
            (sqlite.override { interactive = true; })
            (builtins.getAttr systemString RNMC.defaultPackage)
          ];
        };

    in {
      devShell = {
        x86_64-linux = genericDevShell "x86_64-linux" false;
        x86_64-darwin = genericDevShell "x86_64-darwin" false;
      };

      defaultPackage = {
        x86_64-linux = NanoParticleTools "x86_64-linux";
        x86_64-darwin = NanoParticleTools "x86_64-darwin";
      };

      checks = {
        x86_64-linux.tests = NanoParticleTools "x86_64-linux";
        x86_64-darwin.tests = NanoParticleTools "x86_64-darwin";
      };
    };

}
