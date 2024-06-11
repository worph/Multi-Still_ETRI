# Define variables
$imageName = "my-python-dev"
$containerName = "python-dev-container"

# Get the current working directory
$cwd = (Get-Location).Path

# Build the Docker image
docker build -t $imageName .

# Check if the container is already running and stop it
$existingContainer = docker ps -aq --filter "name=$containerName"
if ($existingContainer) {
    docker stop $existingContainer
    docker rm $existingContainer
}

# Run the Docker container with the current directory mounted to /app
docker run -it --gpus all --name $containerName -v "${cwd}:/app" $imageName /bin/bash
