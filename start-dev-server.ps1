$imageName = "nlp-server"
$containerName = "nlp-server-dev"

# Build the Docker image
docker build -t $imageName .

# Check if the container is already running and stop it
$existingContainer = docker ps -aq --filter "name=$containerName"
if ($existingContainer) {
    docker stop $existingContainer
    docker rm $existingContainer
}

docker run --gpus all --name $containerName -p "5000:5000" $imageName
