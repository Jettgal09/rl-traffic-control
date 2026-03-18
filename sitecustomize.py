# sitecustomize.py
# Adds project root to Python path so all modules are importable.

import sys
import os

# Get the directory this file lives in (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add it to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
