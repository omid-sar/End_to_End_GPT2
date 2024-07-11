# GPT2 & Cloud Deployment & CI/CD pipeline 


## Overview

Overview
Overview
Overview
## Getting Started

### Prerequisites
- Python 3.11

### Installation Steps
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/omid-sar/End_to_End_GPT2.git
   ```
    ```bash
    cd End_to_End_GPT2
    ```
2. **Create and Activate a Conda Environment**: 

    ```bash
    pip install -r requirements.txt
    ```
   ```bash
   conda activate gpt2

3. **Install the Requirements**: 
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**: 
   ```bash
   python app.py
   ```

   After running, access the application via your local host and specified port.

## Contact
- **Author**: Omid Sar
- **Email**: [mr.omid.sardari@gmail.com](mailto:mr.omid.sardari@gmail.com)

---


# Connect to Lambda Labs GPU with VSCode SSH


## Configure SSH Connection
1. **Open VSCode Command Palette**: Press `CMD + SHIFT + P`.
2. **Add NEW HOST**
```bash
Host lambda-gpu
    HostName 104.171.202.100
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
```
2. If already create a new host **Edit SSH Config**: Choose `SSH: Open SSH Config` and update the `HostName` with your Lambda Labs instance IP `104.171.202.100`.

## Connect to Your Instance
- **Connect**: Return to the Command Palette, select `SSH: Connect to Host`, and choose your configured host.

## File Transfer Using SCP
Transfer files efficiently from your local machine to your Lambda Labs instance , If you navigate to the directory containing the project folder from the terminal, and running the followind command in your local terminal:

```bash
tar --exclude=".*" -czf - ./ 2>/dev/null | ssh lambda-gpu "mkdir -p ~/gpt && tar -xzf - -C ~/gpt"
tar -czvf - ./ 2>/dev/null | ssh lambda-gpu "mkdir -p ~/gpt && tar -xzvf - -C ~/gpt"  
 ```
 Details
1. `tar`: The command to create or extract archive files.

2. `--exclude=".*"`: Excludes all files and directories that start with a dot (hidden files in Unix-like systems).

3. `-czf -`:
   - `-c`: Create a new archive
   - `-z`: Compress the archive using gzip
   - `-f -`: Use stdout as the archive file ('-' means stdout)

4. `./`: The directory to archive (current directory in this case)

5. `2>/dev/null`: Redirects standard error (stderr) to /dev/null, effectively silencing any error messages or warnings

6. `|`: Pipes the output of the tar command to the input of the ssh command

7. `ssh lambda-gpu`: Connects to the remote server named 'lambda-gpu' (as defined in your SSH config)

8. `"mkdir -p ~/gpt && tar -xzf - -C ~/gpt"`: The command to be executed on the remote server:
   - `mkdir -p ~/gpt`: Creates the ~/gpt directory if it doesn't exist
   - `&&`: Executes the next command only if the previous one succeeds
   - `tar -xzf - -C ~/gpt`:
     - `-x`: Extract files from an archive
     - `-z`: The archive is compressed with gzip
     - `-f -`: Read the archive from stdin ('-' means stdin)
     - `-C ~/gpt`: Change to the ~/gpt directory before extracting

In summary, this command does the following:
1. Creates a compressed tar archive of the current directory, excluding hidden files.
2. Silences any warnings or errors from the tar command.
3. Pipes this archive directly to the ssh command.
4. Connects to the remote server.
5. On the remote server, it creates a ~/gpt directory if it doesn't exist.
6. Extracts the received archive into the ~/gpt directory on the remote server.
## Setting Up VSCode Extensions via SSH

To streamline your VSCode setup on remote servers through SSH, use the provided script to install essential extensions. Execute these commands in your terminal:

```bash
# Make the script executable
chmod +x install_vscode_extensions.sh

# Run the script to install extensions
./install_vscode_extensions.sh
```
This script ensures that your remote VSCode environment mirrors the robustness of your local setup with necessary extensions pre-installed. Modify install_vscode_extensions.sh directly in the repository to customize which extensions are installed.


## Git configuration on Lambda Labs Terminal :

```bash
git config --global user.name "omid-sar"
git config --global user.email "mr.omid.sardari@gmail.com"
 ```

## Special Features

- **Transformer Model Implementation**: Built from the ground up based on the influential "Attention is All You Need" paper, this project utilizes PyTorch to create a powerful and efficient translation model.

- **Modular Pipeline Design**: The project adopts a modular approach, with the pipeline segmented into independent stages (Data Ingestion, Validation, Transformation, Model Training, Evaluation, and Inference). This design facilitates easier debugging, iterative development, and scalability.

- **MLOps Integration**: Demonstrates comprehensive MLOps practices by integrating continuous integration and continuous delivery (CI/CD) pipelines using GitHub Actions. This ensures that any changes to the codebase automatically trigger workflows for testing, building, and deploying the application, maintaining the project in a release-ready state.

- **Cloud Deployment and Containerization**: The model is containerized using Docker, making it platform-independent and easily deployable on cloud services like Amazon EC2. This approach underscores the project's readiness for real-world applications and ease of use across different environments.

- **Interactive Model Access**: Utilizing Gradio for creating a user-friendly web interface, the project allows easy access to the translation model through a simple interface, enabling users to experience the model's capabilities directly.


## Contact
- **Author**: Omid Sar
- **Email**: [mr.omid.sardari@gmail.com](mailto:mr.omid.sardari@gmail.com)

---

# AWS CI/CD Deployment with Github Actions

## Overview
This guide provides a comprehensive walkthrough for deploying a Dockerized application on AWS EC2 using Github Actions for continuous integration and continuous deployment (CI/CD).

## Prerequisites
- AWS Account
- Github Account

## Steps

### 1. AWS Console Preparation
   - **Login**: Ensure you are logged into your AWS console.
   - **Create IAM User**: Ensure the user has the following policies:
     - `AmazonEC2ContainerRegistryFullAccess`
     - `AmazonEC2FullAccess`
   - **Create ECR Repository**:

### 2. EC2 Instance Setup
   - **Create an EC2 Instance**: Preferably Ubuntu.
   - **Install Docker on EC2**: 
     - Optional: Update and upgrade the system.
       ```bash
       sudo apt-get update -y
       ```
       ```bash
       sudo apt-get upgrade
       ```

     - Required: Install Docker.
       ```bash
       curl -fsSL https://get.docker.com -o get-docker.sh
       ```
       ```bash
       sudo sh get-docker.sh
       ```
       ```bash
       sudo usermod -aG docker ubuntu
       ```
       ```bash
       newgrp docker
       ```

### 3. Configure Self-hosted Runner on Github
   - Navigate to your repository's settings.
   - Go to Actions > Runners.
   - Click "New self-hosted runner" and follow the instructions.

### 4. Set Up Github Secrets
   - Navigate to your repository's settings.
   - Go to Secrets and add the following:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`
     - `AWS_REGION`
     - `AWS_ECR_LOGIN_URI`
     - `ECR_REPOSITORY_NAME`

## Deployment Flow
1. **Build Docker Image**: Locally or in CI/CD pipeline.
2. **Push Docker Image to ECR**: Use AWS CLI or Github Actions.
3. **Launch EC2 Instance**: Ensure it has Docker installed.
4. **Pull Docker Image on EC2**: Use AWS CLI.
5. **Run Docker Container on EC2**: Start your application.
