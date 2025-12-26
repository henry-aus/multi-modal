#!/bin/bash
echo "Setting up Home Robot Python 3.12 environment..."

# Create virtual environment with Python 3.12
/opt/homebrew/bin/python3.12 -m venv home_robot_env

# Activate environment
source home_robot_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source home_robot_env/bin/activate"