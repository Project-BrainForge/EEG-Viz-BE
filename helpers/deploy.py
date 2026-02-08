import os
import subprocess


def build_and_deploy():
    """
    Builds and deploys the application using Docker Compose.
    """
    compose_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docker-compose.yml')
    if not os.path.exists(compose_file):
        raise FileNotFoundError(f"docker-compose.yml not found at {compose_file}")

    print("Building Docker images...")
    build_cmd = ["docker-compose", "-f", compose_file, "build"]
    subprocess.run(build_cmd, check=True)

    print("Starting services with docker-compose up...")
    up_cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
    subprocess.run(up_cmd, check=True)
    print("Application deployed and running.")


def stop_services():
    """
    Stops the running Docker Compose services.
    """
    compose_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docker-compose.yml')
    if not os.path.exists(compose_file):
        raise FileNotFoundError(f"docker-compose.yml not found at {compose_file}")

    print("Stopping services with docker-compose down...")
    down_cmd = ["docker-compose", "-f", compose_file, "down"]
    subprocess.run(down_cmd, check=True)
    print("Services stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deploy or stop the application using Docker Compose.")
    parser.add_argument("action", choices=["deploy", "stop"], help="Action to perform: deploy or stop")
    args = parser.parse_args()

    if args.action == "deploy":
        build_and_deploy()
    elif args.action == "stop":
        stop_services()
