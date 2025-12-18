# RiverSeg

**RiverSeg** is a Python pipeline for **segmenting river surface water from satellite imagery** using deep learning, with an explicit focus on **topologically consistent predictions**.

The project aims to produce **robust binary river masks** that preserve river continuity and structure, making them suitable as input for downstream vectorization or graph-based analyses.

This repository contains:
- A reproducible **preprocessing pipeline**
- A **trained segmentation model** for inference
- Scripts and examples to run predictions on sample data

> In a companion repository, **mask2graph**, tools to convert binary masks into graph representations can be found.


While this repository focuses on mask prediction, the design choices (loss functions, preprocessing, postprocessing) are informed by downstream **topological requirements**.

---

## Features
- Satellite image preprocessing (band selection, normalization, tiling)
- Deep learning–based river segmentation (U-Net–style architecture)
- Inference on image patches or tiles
- Output of binary river masks (GeoTIFF / PNG)
- Simple segmentation metrics (IoU, Dice)

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/mort607e/riverseg.git
cd riverseg
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```


### 3. Run preprocessing on sample data
```bash
python scripts/preprocess_sample.py
```

### 4. Run inference
```bash
python scripts/predict_sample.py
```

This will generate:
- A binary river mask
- A visualization overlay in the `outputs/` directory

---

## Example Output
*(Add images here once available)*

- Input satellite patch  
- Predicted river mask  
- Mask overlay on input image  

---

## Model
The segmentation model is based on a **U-Net–style architecture**, trained on satellite imagery with **pseudo-labels derived from spectral water indices**.

- **Input:** preprocessed satellite image patches  
- **Output:** binary river mask  
- **Loss functions:** Dice / BCE (see `TRAINING.md`)

Pretrained weights are provided in the `weights/` directory (or via a download script).

---

## Evaluation
Basic segmentation performance is reported using:
- Intersection over Union (IoU)
- Dice coefficient
- Precision and Recall

For Connectivity and topology-based evaluation, Number of connected components (NCC) are used as a simple proxy metric.

---

## Note
- Trained using pseudo-labels derived from spectral indices (not manual annotations)
- Performance varies across environments (e.g. ice, wetlands, cloud cover)

---

## Related Work
- **Mask → graph conversion:** see the companion repository **mask2graph**  
  *(link will be added)*

---

## Citation
If you use this code in academic work, please cite:

> M. M. Christensen,  
> *River segmentation and graph extraction from multimodal satellite imagery*,  
> MSc Thesis, Technical University of Denmark (DTU), 2025.

---

## License
This project is released under the **MIT License**.
