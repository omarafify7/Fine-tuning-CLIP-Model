import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPTextModel

class CLIPModel(nn.Module):
    """
    CLIP Model with ResNet50 Image Encoder and Frozen CLIP Text Encoder.
    """
    def __init__(self, embed_dim=512, hidden_dim=2048, text_model_name="openai/clip-vit-base-patch32"):
        super(CLIPModel, self).__init__()
        
        # ======================================================================
        # Image Encoder (ResNet50)
        # ======================================================================
        resnet = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        # ResNet50 structure: ... -> avgpool -> fc
        # We want the output of avgpool, which is (batch, 2048, 1, 1)
        # So we take everything up to fc
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection Head for Image
        self.image_projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # ======================================================================
        # Text Encoder (Frozen CLIP)
        # ======================================================================
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name, use_safetensors=True)
        
        # Freeze Text Encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, text_inputs):
        """
        Args:
            images (torch.Tensor): (batch, 3, 224, 224)
            text_inputs (torch.Tensor): (batch, 77) - Input IDs
                                        OR (batch, embed_dim) - Precomputed Embeddings
        
        Returns:
            image_features (torch.Tensor): (batch, embed_dim)
            text_features (torch.Tensor): (batch, embed_dim)
        """
        # Image Encoding
        # image_encoder output: (batch, 2048, 1, 1)
        image_features = self.image_encoder(images)
        image_features = torch.flatten(image_features, 1) # (batch, 2048)
        image_features = self.image_projection(image_features) # (batch, embed_dim)
        
        # Text Encoding
        if text_inputs.dim() == 2 and text_inputs.shape[1] == 77: # Input IDs
            with torch.no_grad():
                text_outputs = self.text_encoder(text_inputs)
                text_features = text_outputs.pooler_output # (batch, embed_dim)
        else: # Precomputed Embeddings
            text_features = text_inputs
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features
