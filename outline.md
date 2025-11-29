# ELEC 475 Lab 4: Fine-tuning CLIP - Actionable Outline

## 1. Introduction & Goal
Implement and train a CLIP-style model using a frozen text encoder (OpenAI CLIP) and a trainable image encoder (ResNet50) on the MS COCO 2014 dataset.

## 2. Task Breakdown

### 2.1 Dataset Preparation (MS COCO)
- [ ] **Download Data**:
    - [ ] `train2014/` images
    - [ ] `val2014/` images
    - [ ] `captions_train2014.json`
    - [ ] `captions_val2014.json`
- [ ] **Image Preprocessing**:
    - [ ] Resize to **224x224**
    - [ ] Normalize using CLIP stats:
        - Mean: `(0.48145466, 0.4578275, 0.40821073)`
        - Std: `(0.26862954, 0.26130258, 0.27577711)`
- [ ] **Text Preprocessing**:
    - [ ] Use pretrained CLIP text encoder (`openai/clip-vit-base-patch32`).
    - [ ] **Optimization Hint**: Pre-encode all captions and save embeddings + image_ids to `.pt` cache files to save time/memory.
- [ ] **Verification**:
    - [ ] Display random image-caption pairs to verify integrity.

### 2.2 Model Design
- [ ] **Image Encoder**:
    - [ ] ResNet50 (pretrained on ImageNet).
    - [ ] **Projection Head**: Two linear layers with GELU activation mapping to **512-dim** embedding space.
- [ ] **Text Encoder**:
    - [ ] Frozen pretrained Transformer (OpenAI CLIP).
    - [ ] **Important**: Freeze all parameters; only train image encoder + projection head.

### 2.3 Training
- [ ] **Loss Function**: Implement **InfoNCE loss**.
- [ ] **Training Loop**:
    - [ ] Maximize alignment between paired images and captions.
    - [ ] Experiment with LR, optimizers, batch sizes.
- [ ] **Logging**:
    - [ ] Track Training & Validation Loss.
    - [ ] Track Total Training Time & Hardware used.
    - [ ] Note any issues (divergence, instability).

### 2.4 Evaluation & Visualization
- [ ] **Metrics (Validation Set)**:
    - [ ] Compute Cosine Similarity Matrix.
    - [ ] Calculate **Recall@1, Recall@5, Recall@10** for:
        - [ ] Image-to-Text Retrieval
        - [ ] Text-to-Image Retrieval
- [ ] **Visualization**:
    - [ ] **Text-to-Image**: Query "sport" -> Show top-5 images.
    - [ ] **Zero-Shot Classification**: Image + classes ["person", "animal", "landscape"] -> Classify.

### 2.5 Modifications (Ablation Study)
- [ ] Implement **at least two** modifications to improve accuracy (e.g., normalization layers, architecture changes, data augmentation).
- [ ] Repeat evaluation (Section 2.4) for each modification.
- [ ] Compare results to baseline.

## 3. Deliverables

### 3.1 Code Structure
- [ ] Compressed project directory containing:
    - [ ] All `.py` scripts (model, dataset, training).
    - [ ] `Train.txt`: Command to start training.
    - [ ] `Test.txt`: Command to start evaluation.
    - [ ] Generated qualitative results (images).

### 3.2 Report Sections
- [ ] **Introduction**: Motivation & CLIP structure.
- [ ] **Methodology**: Model & training design.
- [ ] **Results**: Quantitative metrics (Recall@K, Loss plots).
- [ ] **Discussion**: Interpret results, analyze behavior.
- [ ] **Conclusion**: Reflection & improvements.
- [ ] **Appendix**: Code snippets, diagrams.
- [ ] **LLM Reflection**: How LLMs were used, example queries, link to conversation.

## 4. Evaluation Rubric (Total: 16 Marks)
- Implementation & Model Design: 4
- Training: 1
- Modifications & Ablations: 2
- Evaluation: 2
- Report: 6
- Submission Format: 1
