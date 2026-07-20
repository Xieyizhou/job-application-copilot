#!/bin/zsh

echo "Starting Job Application Toolkit..."
echo "-----------------------------------"

# Go to the folder where this command file is located
cd "$(dirname "$0")"

# Check virtual environment
if [ ! -d ".venv" ]; then
  echo "Error: .venv folder not found."
  echo "Please run: python3 -m venv .venv"
  exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check dashboard launcher
if [ ! -f "run_dashboard.py" ]; then
  echo "Error: run_dashboard.py not found."
  exit 1
fi

# Start Streamlit dashboard
python run_dashboard.py
