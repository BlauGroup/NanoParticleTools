{
  description = "Multi node reaction network generator";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-21.05;
    RNMC.url = github:BlauGroup/RNMC;
  };

  outputs = { self, nixpkgs, RNMC }:
    let
      genericDevShell = systemString:
        with import nixpkgs { system = systemString; };
        mkShell {
          buildInputs = [
            (python38.withPackages (ps: [ ]))
            sqlitebrowser
          ];
        };

    in {
      devShell = {
        x86_64-linux = genericDevShell "x86_64-linux";
        x86_64-darwin = genericDevShell "x86_64-darwin";
      };
    };
}
