# GPT2 & Cloud Deployment & CI/CD pipeline 


## Overview

Overview
Overview
Overview


## Getting Started

### Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.8

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
Transfer files efficiently from your local machine to your Lambda Labs instance using the SCP command, If you navigate to the directory containing the project folder from the terminal, the second of "cache files are excluded":

```bash
scp -i ~/.ssh/id_ed25519 -r ./ ubuntu@104.171.203.119:~/
rsync -avz --exclude='.cache/' --exclude='*/cache/' -e "ssh -i ~/.ssh/id_ed25519" ./ ubuntu@104.171.203.119:~/gpt
 ```

ey Details

	•	-i ~/.ssh/id_ed25519: Specifies the SSH private key for authentication.
	•	-r: Enables recursive copying, necessary for directories.
	•	ubuntu@XXX.XXX.XXX.XXX:~/: Target directory on the remote instance.

Ensure the SSH key permissions are correctly set (chmod 600) to prevent any connection issues.

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
