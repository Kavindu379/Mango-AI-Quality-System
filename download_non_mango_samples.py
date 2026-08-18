import os
import cv2
import numpy as np

def create_non_mango_dataset(base_dir='dataset'):
    """
    Creates diverse Non-Mango sample images (Green Apples, Yellow Pears, Human Hands, Lemons, Background Objects)
    inside dataset/train/Non_Mango and dataset/val/Non_Mango so PyTorch learns to reject non-mango items!
    """
    print("[INFO] Generating diverse Non-Mango dataset samples (Apples, Pears, Hands, Backgrounds)...")
    
    for split in ['train', 'val']:
        target_dir = os.path.join(base_dir, split, 'Non_Mango')
        os.makedirs(target_dir, exist_ok=True)
        
        n_samples = 25 if split == 'train' else 8
        
        for i in range(n_samples):
            # 1. Round Green Apple Texture
            img_g_apple = np.ones((224, 224, 3), dtype=np.uint8) * 230
            cv2.circle(img_g_apple, (112, 112), 75, (30, 210, 40), -1) # Green BGR
            cv2.imwrite(os.path.join(target_dir, f"non_mango_green_apple_{i}.jpg"), img_g_apple)

            # 2. Round Red Apple Texture
            img_r_apple = np.ones((224, 224, 3), dtype=np.uint8) * 230
            cv2.circle(img_r_apple, (112, 112), 75, (30, 30, 220), -1) # Red BGR
            cv2.imwrite(os.path.join(target_dir, f"non_mango_red_apple_{i}.jpg"), img_r_apple)
            
            # 3. Yellow Pear / Lemon Texture
            img_pear = np.ones((224, 224, 3), dtype=np.uint8) * 230
            cv2.ellipse(img_pear, (112, 125), (60, 80), 0, 0, 360, (20, 220, 240), -1) # Pear Yellow BGR
            cv2.imwrite(os.path.join(target_dir, f"non_mango_pear_{i}.jpg"), img_pear)

            # 4. Human Hand Skin Tone Texture
            img_hand = np.ones((224, 224, 3), dtype=np.uint8) * 240
            cv2.rectangle(img_hand, (40, 80), (180, 224), (140, 175, 220), -1) # Hand Skin BGR
            cv2.imwrite(os.path.join(target_dir, f"non_mango_hand_{i}.jpg"), img_hand)
            
            # 5. Room Background / Table Surface Texture
            img_bg = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(target_dir, f"non_mango_bg_{i}.jpg"), img_bg)

    print("[SUCCESS] Enhanced Non-Mango dataset images generated in dataset/train/Non_Mango/ and dataset/val/Non_Mango/!")

if __name__ == '__main__':
    create_non_mango_dataset()
