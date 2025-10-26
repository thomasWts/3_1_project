import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
print("Model loaded successfully!")

# 定义Pokemon类别和对应的文本描述
pokemon_classes = {
    "pikachu": ["a yellow electric mouse pokemon", "pikachu", "an electric type pokemon"],
    "bulbasaur": ["a green grass pokemon with a bulb", "bulbasaur", "a grass type pokemon"],
    "charmander": ["an orange fire lizard pokemon", "charmander", "a fire type pokemon"],
    "squirtle": ["a blue water turtle pokemon", "squirtle", "a water type pokemon"],
    "mewtwo": ["a purple psychic pokemon", "mewtwo", "a legendary psychic pokemon"]
}

# 测试每个Pokemon类别
base_path = "g:/Thomas/3_1_project/data/pokemon"
print("\n" + "="*70)
for pokemon_name in pokemon_classes.keys():
    pokemon_dir = os.path.join(base_path, pokemon_name)
    
    # 找到第一个可用的图片文件
    image_file = None
    for ext in ['.jpg', '.png', '.jpeg']:
        test_file = os.path.join(pokemon_dir, f"00000000{ext}")
        if os.path.exists(test_file):
            image_file = test_file
            break
    
    if image_file:
        print(f"\n📸 Testing: {pokemon_name.upper()}")
        print(f"   Image: {image_file}")
        
        # 加载图片
        image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        
        # 为当前Pokemon准备所有可能的文本标签
        all_labels = []
        for labels_list in pokemon_classes.values():
            all_labels.extend(labels_list)
        
        text = clip.tokenize(all_labels).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # 找出前3个最可能的标签
        top3_indices = probs[0].argsort()[-3:][::-1]
        print(f"\n   Top 3 predictions:")
        for i, idx in enumerate(top3_indices, 1):
            print(f"   {i}. {all_labels[idx]}: {probs[0][idx]:.4f}")
        
        print("-" * 70)

print("\n✅ All tests completed!")
