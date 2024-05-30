from testcontainers.compose import DockerCompose
import requests
import time


def test_server_response():
    compose = DockerCompose(
        "./"
    )  # Assuming docker-compose.yml is in the same directory
    print(compose)
    
    with compose:
        # Wait a few seconds for the server to be up
        time.sleep(5)

        # Docker Compose typically uses the service name as the hostname
        host,port = compose.get_service_host_and_port("hypha",9000)

        # Construct the URL to test
        url = f"http://{host}:{port}"

        response = requests.get(url)
        if response.status_code == 200:
            print("Server responded successfully!")
        else:
            print(f"Server responded with status code: {response.status_code}")
        
