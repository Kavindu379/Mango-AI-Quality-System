import os
import urllib.request
import urllib.parse
import json
import time

def download_wiki():
    # 50 guaranteed topics that have clear main images on Wikipedia
    topics = [
        "Apple", "Banana", "Orange (fruit)", "Lemon", "Lime (fruit)", "Avocado", 
        "Tomato", "Potato", "Onion", "Garlic", "Carrot", "Cucumber", "Broccoli",
        "Coffee cup", "Mug", "Bottle", "Glass (drinkware)", "Plate", "Bowl",
        "Tennis ball", "Baseball", "Basketball", "Soccer ball", "Golf ball",
        "Human hand", "Finger", "Human eye", "Ear", "Nose", "Mouth",
        "Computer mouse", "Computer keyboard", "Laptop", "Mobile phone", "Tablet computer",
        "Shoe", "Sock", "Hat", "Shirt", "Pants",
        "Cat", "Dog", "Bird", "Fish", "Tree",
        "Bread", "Cheese", "Pizza", "Hamburger", "Sushi"
    ]
    
    print("[INFO] Fetching 50 object images from Wikipedia...")
    
    train_dir = os.path.join('dataset', 'train', 'Non_Mango')
    val_dir = os.path.join('dataset', 'val', 'Non_Mango')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    count = 0
    for topic in topics:
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(topic)}&prop=pageimages&format=json&pithumbsize=400"
            req = urllib.request.Request(url, headers={'User-Agent': 'MangoAI/1.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                pages = data['query']['pages']
                for page_id in pages:
                    if 'thumbnail' in pages[page_id]:
                        img_url = pages[page_id]['thumbnail']['source']
                        
                        target_dir = train_dir if count % 5 != 0 else val_dir
                        file_path = os.path.join(target_dir, f"non_mango_wiki_{count}.jpg")
                        
                        req2 = urllib.request.Request(img_url, headers={'User-Agent': 'MangoAI/1.0'})
                        with urllib.request.urlopen(req2) as resp2, open(file_path, 'wb') as out_file:
                            out_file.write(resp2.read())
                        print(f"[{count+1}/50] Downloaded {topic}")
                        count += 1
        except Exception as e:
            pass
        time.sleep(0.1)
        
    print(f"\n[SUCCESS] Downloaded {count} real object images.")

if __name__ == '__main__':
    download_wiki()
