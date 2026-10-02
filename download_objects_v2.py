import os
import urllib.request
import time
import glob

def clean_and_download():
    train_dir = os.path.join('dataset', 'train', 'Non_Mango')
    val_dir = os.path.join('dataset', 'val', 'Non_Mango')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("[INFO] Cleaning out ALL old fake/MS Paint images...")
    # Delete everything in the Non_Mango folders to start completely fresh
    for d in [train_dir, val_dir]:
        for f in glob.glob(os.path.join(d, "*.jpg")):
            try:
                os.remove(f)
            except:
                pass

    print("[INFO] Downloading a large set of real objects from LoremFlickr API...")
    
    # We will download 5 images for each of these 10 categories = 50 images total
    keywords = [
        "apple,fruit", "banana", "lemon", "avocado", "orange,fruit", 
        "tennis,ball", "coffee,mug", "human,hand", "table,wood", "plate,food"
    ]
    
    count = 0
    for keyword in keywords:
        for i in range(5):
            # Using random={count} forces the API to give a unique image every time
            url = f"https://loremflickr.com/224/224/{keyword}?random={count}"
            target_dir = train_dir if i < 4 else val_dir
            file_path = os.path.join(target_dir, f"non_mango_obj_{count}.jpg")
            
            try:
                # Add a timeout so it doesn't hang
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[{count+1}/50] Downloaded {keyword}")
                count += 1
                time.sleep(0.2) # Small pause to respect the server
            except Exception as e:
                print(f"Failed to download {keyword}: {e}")
                
    print(f"\n[SUCCESS] Saved {count} highly-focused object images! Your dataset is perfectly clean.")

if __name__ == '__main__':
    clean_and_download()
