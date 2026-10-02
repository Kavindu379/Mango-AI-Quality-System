import json

def update_notebook():
    file_path = 'Mango_Quality_CNN_Training.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update cell 4 to add the '-o' overwrite flag
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('uploaded = files.upload()' in line for line in cell.get('source', [])):
            cell['source'] = [
                "import os\n",
                "import shutil\n",
                "from google.colab import files\n",
                "\n",
                "print('Click below to upload your mango dataset ZIP file:')\n",
                "uploaded = files.upload()\n",
                "\n",
                "for filename in uploaded.keys():\n",
                "    if filename.endswith('.zip'):\n",
                "        print(f'Extracting {filename} using native unzip...')\n",
                "        # Added -o flag to automatically overwrite without freezing the notebook!\n",
                "        !unzip -q -o \"{filename}\" -d temp_extract/\n",
                "        print(f'Unzipped {filename} into temporary folder.')\n",
                "\n",
                "# --- Auto-Detect Nested Folders ---\n",
                "extract_root = 'temp_extract'\n",
                "nested_dirs = [d for d in os.listdir(extract_root) if not d.startswith('__MACOSX')]\n",
                "if len(nested_dirs) == 1 and os.path.isdir(os.path.join(extract_root, nested_dirs[0])):\n",
                "    source_dir = os.path.join(extract_root, nested_dirs[0])\n",
                "else:\n",
                "    source_dir = extract_root\n",
                "\n",
                "if os.path.exists('dataset'):\n",
                "    shutil.rmtree('dataset')\n",
                "shutil.move(source_dir, 'dataset')\n",
                "if os.path.exists('temp_extract'):\n",
                "    shutil.rmtree('temp_extract')\n",
                "\n",
                "print(\"\\n[SUCCESS] Dataset successfully mapped to 'dataset/' directory!\")\n",
                "\n",
                "CLASS_NAMES = ['Grade_A_Ripe', 'Grade_B_Unripe', 'Grade_C_Overripe', 'Non_Mango']\n",
                "for split in ['train', 'val']:\n",
                "    for c in CLASS_NAMES:\n",
                "        p = os.path.join('dataset', split, c)\n",
                "        if os.path.exists(p):\n",
                "            n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))])\n",
                "            print(f'  dataset/{split}/{c}: {n} images')"
            ]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_notebook()
