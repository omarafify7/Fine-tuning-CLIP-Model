import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from dataset import CocoDataset, get_transforms
from model import CLIPModel
import util

def compute_embeddings(model, dataloader, device):
    model.eval()
    image_embeddings = []
    text_embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for images, text_inputs in tqdm(dataloader):
            images = images.to(device)
            text_inputs = text_inputs.to(device)
            
            img_emb, txt_emb = model(images, text_inputs)
            
            image_embeddings.append(img_emb.cpu())
            text_embeddings.append(txt_emb.cpu())
            
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    return image_embeddings, text_embeddings

def calculate_recall(image_embeddings, text_embeddings, k_values=[1, 5, 10]):
    # Cosine similarity: (num_images, num_texts)
    # Note: In COCO, there are 5 captions per image.
    # So we have N images and 5N captions.
    # But our dataloader returns pairs.
    # If we use the validation set as is, we have duplicate images.
    # Standard evaluation: Unique images, and for each image, 5 captions.
    # Or for each caption, find the image.
    
    # Let's assume we want Image-to-Text (I2T) and Text-to-Image (T2I).
    # Similarity matrix: (num_samples, num_samples) if we use the dataloader pairs.
    # But since images are duplicated (5 times), we should be careful.
    # However, for simplicity in this lab, let's just compute pairwise similarity 
    # between the validation set pairs (which effectively treats each caption-image pair as unique).
    # A better way is to group by image_id, but that requires more logic.
    # Given the outline says "Compute Cosine Similarity Matrix", let's do N x N.
    
    print("Computing similarity matrix...")
    # Normalize just in case (model should output normalized, but safe to redo)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    similarity = image_embeddings @ text_embeddings.t()
    
    num_samples = similarity.shape[0]
    
    # Image-to-Text Retrieval
    # For each image, find the correct caption.
    # The correct caption index for image i is i (since they are paired in dataloader).
    # But wait, if we have duplicate images, image i and image j might be the same image with different captions.
    # If we treat them as distinct pairs, then diagonal is the ground truth.
    
    print("Calculating Recall metrics (assuming diagonal is ground truth)...")
    
    # I2T Recall
    # For each row i, is i in top K?
    topk_indices = torch.topk(similarity, k=max(k_values), dim=1).indices
    i2t_recalls = {k: 0.0 for k in k_values}
    
    for i in range(num_samples):
        for k in k_values:
            if i in topk_indices[i, :k]:
                i2t_recalls[k] += 1
                
    for k in k_values:
        i2t_recalls[k] /= num_samples
        print(f"I2T Recall@{k}: {i2t_recalls[k]:.4f}")
        
    # T2I Recall
    # For each col j, is j in top K?
    topk_indices_t = torch.topk(similarity, k=max(k_values), dim=0).indices # (k, num_samples)
    t2i_recalls = {k: 0.0 for k in k_values}
    
    for j in range(num_samples):
        for k in k_values:
            if j in topk_indices_t[:k, j]:
                t2i_recalls[k] += 1
                
    for k in k_values:
        t2i_recalls[k] /= num_samples
        print(f"T2I Recall@{k}: {t2i_recalls[k]:.4f}")
        
    return i2t_recalls, t2i_recalls

def visualize_text_to_image(model, dataset, device, query_text="a person playing sports", top_k=5):
    print(f"Visualizing Text-to-Image for query: '{query_text}'")
    model.eval()
    
    # Encode text
    if isinstance(dataset, torch.utils.data.Subset):
        tokenizer = dataset.dataset.tokenizer
    else:
        tokenizer = dataset.tokenizer
    inputs = tokenizer(query_text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        text_outputs = model.text_encoder(input_ids)
        text_feature = text_outputs.pooler_output
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
    # We need image embeddings for the whole dataset (or a subset)
    # This is expensive to recompute. In a real app, we'd cache them.
    # For this script, let's assume we pass precomputed image embeddings or compute them on a subset.
    # Let's compute on a random subset of 100 images from dataset.
    
    indices = np.random.choice(len(dataset), 100, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    image_embeddings = []
    images_to_show = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            img_emb, _ = model(images, input_ids.repeat(images.shape[0], 1)) # Dummy text
            image_embeddings.append(img_emb.cpu())
            # Store images for visualization (un-normalize)
            # This is tricky because images are tensors.
            # We can reload them or undo normalization.
            images_to_show.append(images.cpu())
            
    image_embeddings = torch.cat(image_embeddings, dim=0)
    images_to_show = torch.cat(images_to_show, dim=0)
    
    # Similarity
    similarity = text_feature.cpu() @ image_embeddings.t() # (1, N)
    values, indices = torch.topk(similarity, top_k)
    
    # Plot
    fig, axes = plt.subplots(1, top_k, figsize=(15, 3))
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    
    for i, idx in enumerate(indices[0]):
        img_tensor = images_to_show[idx]
        # Un-normalize
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"Score: {values[0, i]:.2f}")
        
    plt.suptitle(f"Query: {query_text}")
    plt.savefig(f"t2i_{query_text.replace(' ', '_')}.png")
    print(f"Saved visualization to t2i_{query_text.replace(' ', '_')}.png")

def zero_shot_classification(model, image_path, classes, device):
    print(f"Zero-shot classification for {image_path} with classes {classes}")
    # Load image
    transform = get_transforms(is_train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tokenize classes
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    text_inputs = tokenizer(classes, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    input_ids = text_inputs['input_ids'].to(device)
    
    with torch.no_grad():
        # Image Feature
        img_emb, _ = model(image_tensor, input_ids[0:1]) # Dummy text
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        # Text Features
        text_outputs = model.text_encoder(input_ids)
        text_features = text_outputs.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Similarity
        similarity = (100.0 * img_emb @ text_features.t()).softmax(dim=-1)
        values, indices = similarity[0].topk(len(classes))
        
    print("\nTop predictions:")
    for value, index in zip(values, indices):
        print(f"{classes[index]}: {value.item():.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../coco2014', help='Root directory of COCO dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    device = util.get_device()
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = CLIPModel().to(device)
    checkpoint = util.load_checkpoint(args.checkpoint, model, device=device)
    
    # Dataset
    val_dir = os.path.join(args.data_dir, 'images', 'val2014')
    val_ann = os.path.join(args.data_dir, 'annotations', 'captions_val2014.json')
    val_dataset = CocoDataset(val_dir, val_ann, transform=get_transforms(is_train=False))
    
    if args.debug:
        indices = torch.arange(100)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Metrics
    image_embeddings, text_embeddings = compute_embeddings(model, val_loader, device)
    calculate_recall(image_embeddings, text_embeddings)
    
    # Visualization
    visualize_text_to_image(model, val_dataset, device, query_text="a person playing baseball")
    
    # Zero-Shot (using a random image from dataset)
    # We need to access the original image path.
    # Dataset returns transformed image.
    # Let's just pick a file from the directory manually for demo.
    # Or use the dataset logic if exposed.
    # For now, skip or use a fixed path if known.
    
if __name__ == "__main__":
    # Need to import CLIPTokenizer here if used in zero_shot
    from transformers import CLIPTokenizer
    main()
