import os
import shutil

def clear_pycache(root_dir="."):
    removed = 0

    for root, dirs, files in os.walk(root_dir):
        # Remove __pycache__ folders
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(pycache_path)
            print(f"Removed folder: {pycache_path}")
            removed += 1

        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")
                removed += 1

    print(f"\nDone ✅ Removed {removed} cache items.")

if __name__ == "__main__":
    clear_pycache()