# fr_ita/config.py
import os

# Get the absolute path to the fr_ita project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
