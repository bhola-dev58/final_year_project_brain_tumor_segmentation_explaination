import shutil
import os

source_dir = "/home/bhola-dev58/.gemini/antigravity-ide/brain/ad5dcb27-061e-4c21-8068-0bbd2a39362c"
dest_dir = "/home/bhola-dev58/colledge project/Brain_Tumor_Project"

images_to_copy = {
    "placeholder_workflow_1783289758962.png": "placeholder_workflow.png",
    "placeholder_roc_1783289781372.png": "placeholder_roc.png",
    "placeholder_segmentation_1783289807610.png": "placeholder_segmentation.png"
}

print("Copying generated research paper images...")
for src_name, dest_name in images_to_copy.items():
    src_path = os.path.join(source_dir, src_name)
    dest_path = os.path.join(dest_dir, dest_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        print(f"Copied: {dest_name}")
    else:
        print(f"Warning: Source file {src_name} not found in artifacts directory.")

print("Done! You can now compile the LaTeX paper with the actual generated images.")
