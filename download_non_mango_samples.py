import os
import urllib.request
import time

def download_real_non_mango_images():
    """
    Downloads random, real-world photographs from the Picsum API 
    (furniture, landscapes, people, objects, backgrounds) 
    to create a robust Non_Mango dataset class.
    """
    print("[INFO] Downloading real-world photos for Non_Mango dataset...")
    
    for split, count in [('train', 60), ('val', 15)]:
        target_dir = os.path.join('dataset', split, 'Non_Mango')
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Downloading {count} images for {split}/Non_Mango...")
        for i in range(count):
            try:
                # Add a unique seed to prevent caching the exact same image
                url = f"https://picsum.photos/seed/mango_{split}_{i}_{time.time()}/224/224"
                file_path = os.path.join(target_dir, f"non_mango_real_{i}.jpg")
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                print(f"Failed to download image {i}: {e}")
                
    print("[SUCCESS] All real-world Non_Mango images downloaded! Your dataset is ready.")

if __name__ == '__main__':
    download_real_non_mango_images()
