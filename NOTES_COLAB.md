import subprocess
import sys

# Run shell commands with real-time output in colab
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    for line in process.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
    process.wait()

# Clone the repository
run_command("git clone https://github.com/klcodingtutor/Attention-CNN-Visualization.git")

# Change to the directory
run_command("cd /content/Attention-CNN-Visualization")

# Pull the latest changes
run_command("git pull")

# Run additional shell scripts
run_command("sh download_cifar10.sh")
run_command("sh multi_view_train_script.sh")

!apt-get update
!apt-get install tree -y
