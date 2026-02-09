import os
import subprocess
import sys

PROJECT_DIR = "/home/ubuntu/EEG-Viz-BE"
SERVICE_NAME = "eegviz"
VENV_PYTHON = "/home/ubuntu/EEG-Viz-BE/venv/bin/python"

git_pulled = False
last_good_commit = ""

def run_command(command, cwd=None, allow_fail=False):
    """ Run a shell command and handle errors unless allow_fail is True """
    print(f"🔧 Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {command}")
        print(f"🔴 Error output:\n{result.stderr}")
        if not allow_fail:
            raise RuntimeError(f"Command failed: {command}")
    else:
        print(f"✅ Output:\n{result.stdout}")
    return result


def get_current_git_commit():
    result = run_command("git rev-parse HEAD", cwd=PROJECT_DIR)
    return result.stdout.strip()

def rollback_git():
    print("⏪ Rolling back to previous commit...")
    run_command(f"git reset --hard {last_good_commit}", cwd=PROJECT_DIR)

def restart_services():
    print(f"🔄 Restarting {SERVICE_NAME} service...")
    run_command("sudo systemctl daemon-reload", allow_fail=True)
    run_command(f"sudo systemctl restart {SERVICE_NAME}", allow_fail=True)

    print(f"📡 Checking {SERVICE_NAME} status...")
    run_command(f"sudo systemctl status {SERVICE_NAME} --no-pager || true", allow_fail=True)


def main():
    global git_pulled, last_good_commit

    print("🚀 Starting deployment...")
    os.chdir(PROJECT_DIR)

    try:
        # Stop FastAPI
        print(f"🛑 Stopping {SERVICE_NAME} service...")
        run_command(f"sudo systemctl stop {SERVICE_NAME}")

        # Store current commit
        last_good_commit = get_current_git_commit()

        # Pull latest changes
        print("🔄 Pulling latest code...")
        run_command("git fetch origin")
        run_command("git reset --hard origin/development")
        git_pulled = True

        # Install dependencies
        print("📦 Installing dependencies...")
        run_command(f"{VENV_PYTHON} -m pip install -r requirements.txt")

        restart_services()

        print("🎉 Deployment complete!")

    except Exception as e:
        print(f"🚨 Deployment failed: {e}")

        if git_pulled:
            print("🔄 Attempting rollback...")
            rollback_git()

            print("📁 Recreating database after rollback...")
            try:
                run_command(f"{VENV_PYTHON} Database/migrate.py")
            except Exception as migration_error:
                print(f"❌ Migration failed during rollback recovery: {migration_error}")
                print("❌ Unable to restore database. Manual intervention needed.")
                sys.exit(1)

        print("🔁 Restarting services to restore state...")
        restart_services()

        print("❌ Deployment rolled back due to error.")
        sys.exit(1)

if __name__ == "__main__":
    main()
