#!/bin/bash

# List of VSCode extensions to install
declare -a extensions=(
    "ms-vscode-remote.remote-ssh"
    "github.copilot"
    "ms-toolsai.jupyter"
    "ms-python.python"
    "ms-vscode.cpptools"
    "vsciot-vscode.vscode-arduino"
    "ms-python.vscode-pylance"
    "ms-kubernetes-tools.vscode-kubernetes-tools"
    "hashicorp.terraform"
    "ms-vscode-remote.remote-ssh-edit"
    "ms-vscode-remote.remote-explorer"
    "ms-vscode.material-icon-theme"
    "redhat.vscode-yaml"
    "equinusocio.vsc-community-material-theme"
    "equinusocio.vsc-material-theme"
    "equinusocio.vsc-material-theme-icons"
)

# Install each extension
for extension in "${extensions[@]}"
do
    code --install-extension $extension
done

echo "All extensions have been installed."