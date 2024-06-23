#!/bin/bash

# List of VSCode extensions to install
declare -a extensions=(
    "amazonwebservices.aws-toolkit-vscode"
    "github.copilot"
    "ms-toolsai.jupyter"
    "ms-python.python"
    "ms-vscode.cpptools"
    "vsciot-vscode.vscode-arduino"
    "ms-python.vscode-pylance"
    "ms-kubernetes-tools.vscode-kubernetes-tools"
    "hashicorp.terraform"
    "ms-vscode.remote-explorer"
    "equinusocio.vsc-material-theme"
    "equinusocio.vsc-material-theme-icons"
)

# Install each extension
for extension in "${extensions[@]}"
do
    code --install-extension $extension
done

echo "All extensions have been installed."