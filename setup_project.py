import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()

def main():
    print("=== Unified Trust Metric System Setup ===")
    
    # 1. Install dependencies
    print("\n[1/3] Installing dependencies...")
    run_command("pip install -r requirements.txt")
    
    # 2. Generate initial dataset
    print("\n[2/3] Generating processed dataset...")
    if os.path.exists('code/fyp dataset.xlsx'):
        run_command("python data_loading.py")
    else:
        print("Warning: 'code/fyp dataset.xlsx' not found. Data generation skipped.")
    
    # 3. Final instructions
    print("\n[3/3] Setup complete!")
    print("\nTo train the model, run:")
    print("   python train.py")
    print("\nTo inspect the model configuration, see:")
    print("   config/config.yaml")

if __name__ == "__main__":
    main()
