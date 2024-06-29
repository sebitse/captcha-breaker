import subprocess

def check_packages(requirements_file):
    try:
        with open(requirements_file, 'r') as file:
            packages = [line.strip().split('==')[0] for line in file if line.strip() and not line.startswith('#')]
        
        for package in packages:
            print(f"\nChecking package: {package}")
            subprocess.run(['pip', 'show', package], check=True)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    check_packages(requirements_file)

