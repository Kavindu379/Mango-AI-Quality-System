import os
import urllib.request
import urllib.parse
import json

def fetch_wiki_image(topic):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(topic)}&prop=pageimages&format=json&pithumbsize=400"
        req = urllib.request.Request(url, headers={'User-Agent': 'MangoAI-Researcher/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data['query']['pages']
            for page_id in pages:
                if 'thumbnail' in pages[page_id]:
                    return pages[page_id]['thumbnail']['source']
    except Exception as e:
        pass
    return None

def download_object_dataset():
    topics = [
        "Apple", "Banana", "Orange (fruit)", "Lemon", "Lime (fruit)", 
        "Avocado", "Tennis ball", "Baseball", "Coffee cup", "Mug", 
        "Human hand", "Potato", "Tomato", "Kiwifruit", "Onion", 
        "Pear", "Peach", "Plum", "Coconut", "Watermelon", 
        "Egg (food)", "Computer mouse", "Scissors", "Pen", "Mobile phone", 
        "Shoe", "Spoon", "Fork", "Table (furniture)", "Plate (dishware)",
        "Papaya", "Pineapple", "Strawberry", "Grape", "Cricket ball"
    ]
    
    print(f"[INFO] Fetching {len(topics)} specific object images from Wikipedia...")
    
    # 1. Clear out the old landscape photos
    for split in ['train', 'val']:
        target_dir = os.path.join('dataset', split, 'Non_Mango')
        os.makedirs(target_dir, exist_ok=True)
        for f in os.listdir(target_dir):
            if f.startswith("non_mango_real"):
                os.remove(os.path.join(target_dir, f))
    
    train_dir = os.path.join('dataset', 'train', 'Non_Mango')
    val_dir = os.path.join('dataset', 'val', 'Non_Mango')
    
    success_count = 0
    for i, topic in enumerate(topics):
        img_url = fetch_wiki_image(topic)
        if img_url:
            target_dir = train_dir if success_count % 5 != 0 else val_dir
            file_path = os.path.join(target_dir, f"non_mango_obj_{success_count}.jpg")
            try:
                req = urllib.request.Request(img_url, headers={'User-Agent': 'MangoAI-Researcher/1.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[{success_count+1}] Downloaded {topic}")
                success_count += 1
            except Exception as e:
                print(f"Failed to save {topic}")
                
    print(f"\n[SUCCESS] Saved {success_count} object-focused images! Old landscapes deleted.")

if __name__ == '__main__':
    download_object_dataset()
