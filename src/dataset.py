import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

class CocoDataset(Dataset):
    """
    COCO Dataset for CLIP training.
    Handles image loading/preprocessing and text tokenization/embedding.
    """
    def __init__(self, root_dir, ann_file, transform=None, tokenizer_name="openai/clip-vit-base-patch32", 
                 max_length=77, use_cached_embeddings=False, cache_dir=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            ann_file (str): Path to the json annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
            tokenizer_name (str): Name of the CLIP tokenizer to use.
            max_length (int): Maximum sequence length for tokenization.
            use_cached_embeddings (bool): If True, load pre-computed embeddings.
            cache_dir (str): Directory to save/load cached embeddings.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.use_cached_embeddings = use_cached_embeddings
        
        # Load annotations
        print(f"Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create a mapping from image_id to filename
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        
        # Filter annotations to only include those with valid images
        self.annotations = [ann for ann in self.coco['annotations'] if ann['image_id'] in self.image_id_to_filename]
        print(f"Loaded {len(self.annotations)} captions.")

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name, use_safetensors=True)

        # Cached Embeddings Logic
        self.cached_embeddings = None
        if self.use_cached_embeddings:
            if cache_dir is None:
                raise ValueError("cache_dir must be provided if use_cached_embeddings is True")
            
            cache_path = os.path.join(cache_dir, f"cached_embeddings_{os.path.basename(ann_file)}.pt")
            if os.path.exists(cache_path):
                print(f"Loading cached embeddings from {cache_path}...")
                self.cached_embeddings = torch.load(cache_path)
            else:
                print(f"Cache not found at {cache_path}. Please run precompute_embeddings first.")
                # Fallback to on-the-fly tokenization if cache missing? 
                # For now, let's assume we want to enforce cache if requested, or we can generate it.
                # But generating it might take time. Let's just warn and disable cache for now or error out.
                raise FileNotFoundError(f"Cached embeddings not found at {cache_path}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']
        filename = self.image_id_to_filename[image_id]
        
        # Load Image
        img_path = os.path.join(self.root_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or skip? Standard practice is hard here without collate_fn handling None.
            # Let's assume data integrity for now or return a zero tensor.
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # Handle Text
        if self.use_cached_embeddings and self.cached_embeddings is not None:
            # Retrieve pre-computed embedding
            # We need to know the index in the cache. 
            # The cache should probably be aligned with self.annotations.
            # If we precomputed based on self.annotations order, we are good.
            text_embedding = self.cached_embeddings[idx]
            return image, text_embedding
        else:
            # On-the-fly tokenization
            inputs = self.tokenizer(
                caption, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            input_ids = inputs['input_ids'].squeeze(0) # (77,)
            # We return input_ids, and the model will encode it.
            # Wait, if we are NOT using cache, we return tokens.
            # If we ARE using cache, we return embeddings.
            # The model's forward pass needs to handle this difference or we need two different training loops/model modes.
            # To keep it simple: Model expects embeddings? Or tokens?
            # The outline says "Frozen pretrained Transformer".
            # If we use cache, we skip the transformer.
            # So the model should probably take `text_inputs` which can be tokens OR embeddings.
            return image, input_ids

def get_transforms(is_train=True):
    """
    Returns the image transformations.
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)), # Resize directly to 224x224 as per outline
            # transforms.RandomHorizontalFlip(), # Optional augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def precompute_embeddings(dataset, model_name="openai/clip-vit-base-patch32", batch_size=32, device='cpu', save_path=None):
    """
    Pre-computes text embeddings for the dataset using a frozen CLIP text encoder.
    """
    print(f"Pre-computing embeddings using {model_name} on {device}...")
    text_encoder = CLIPTextModel.from_pretrained(model_name, use_safetensors=True).to(device)
    text_encoder.eval()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_embeddings = []
    
    with torch.no_grad():
        for i, (images, input_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            # input_ids shape: (batch_size, 77)
            
            # Get embeddings from the text encoder
            # CLIPTextModel output: last_hidden_state, pooler_output
            # We usually use pooler_output for CLIP
            outputs = text_encoder(input_ids)
            pooled_output = outputs.pooler_output # (batch_size, hidden_dim)
            
            all_embeddings.append(pooled_output.cpu())
            
            if i % 100 == 0:
                print(f"Processed {i * batch_size} / {len(dataset)} captions...")
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Finished. Total embeddings: {all_embeddings.shape}")
    
    if save_path:
        torch.save(all_embeddings, save_path)
        print(f"Saved embeddings to {save_path}")
    
    return all_embeddings
