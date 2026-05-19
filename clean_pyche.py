import os
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

deleted = 0

for root, dirs, files in os.walk(ROOT_DIR):
    # Remove __pycache__ folders
    if "__pycache__" in dirs:
        pycache_path = os.path.join(root, "__pycache__")
        shutil.rmtree(pycache_path, ignore_errors=True)
        print(f"Deleted folder: {pycache_path}")
        deleted += 1

    # Remove .pyc files
    for file in files:
        if file.endswith(".pyc"):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            deleted += 1

print(f"\nCleanup complete. Removed {deleted} items.")