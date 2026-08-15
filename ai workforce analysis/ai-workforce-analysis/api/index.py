import sys
import os

# Add base directory to path so server and ai_agent modules are found
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from server import app

# Vercel serverless function entrypoint
