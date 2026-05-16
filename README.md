# Python Scripts for 3D Denoising Model

This directory contains Python scripts for training, preprocessing, and postprocessing 3D U-Net models for image denoising. All scripts were created/expanded and cleaned up in 2025-2026.
It is necessary to remark that these scripts are supposed to be used with scheduling tool slurm.
Directory names need to be changed to fit the environment.
Sample data is not publicly available. 

## Setup

Dependencies will be automatically installed into a python virtual environment.

This model is using python version 3.9.22.

Use `sudo apt install python3.9-distutils` if distutils are not found.


If data is available and the directories are adjusted the model can be executed by using the slurm scripts.

`slurm slurm_jobs/gpu_test.slurm` - Shows the availability of gpu nodes

`slurm slurm_jobs/copy_job.slurm` - Copies files inside the environment

`slurm slurm_jobs/preprocess_job.slurm` - Preprocessing of the image data

`slurm slurm_jobs/cube_model_job.slurm` - Training of the model and data processing

`slurm slurm_jobs/postprocess_job.slurm` - Postprocessing of the image data

Output and error files are found in the main directory. Results can be found in the src directory.


## Scripts Overview

### `cube_model_training.py`
**Purpose:** Train a 3D U-Net model with configurable parameters and optional fine-tuning strategies.

**Key Features:**
- Trains U-Net models on 64×64×64 volumetric data
- Supports multiple fine-tuning approaches (modes 1-3)
- Data augmentation with horizontal/vertical flips and brightness adjustment
- Generates training history plots (loss, MSE, MAE, binary crossentropy, cosine similarity)
- Saves model checkpoints and training metrics as CSV
- GPU optimization with TensorFlow

**Command-line Arguments:**
1. `model_version` - Model version identifier
2. `learning_rate` - Learning rate (e.g., 1e-05)
3. `kernel_size` - Convolutional kernel size
4. `batch_size` - Training batch size
5. `finetuning` - Fine-tuning mode (0=none, 1-3=different strategies)

**Output:**
- Trained model (`.keras`)
- Checkpoint model during training
- Training history (`.csv`)
- Performance plots in `plots/` directory

---

### `preprocessing.py`
**Purpose:** Prepare volumetric data for training by dividing large 3D stacks into overlapping blocks.

**Key Features:**
- Splits 3D image stacks into 64×64×64 blocks with configurable overlap
- Normalizes data to 8-bit unsigned integer format
- Clips values to global min/max statistics across all samples
- Processes both original images and segmentation masks
- Handles directory structure with auto-creation of output folders

**Processing Steps:**
1. Reads 3D stacks from input directory
2. Calculates global min/max values
3. Normalizes and clips data
4. Divides into overlapping cubic blocks
5. Saves processed blocks as TIFF files

**Expected Input Structure:**
```
sample_data/
├── original_images/
└── (CV/merged_cv_v2/ - segmentation masks)
```m can b

**Output:**
```
sample_data/
├── results/U_net/training_data/org_frag_data_3d/
└── results/U_net/training_data/seg_frag_data_3d/
```

---

### `postprocess.py`
**Purpose:** Apply trained model to test data and reconstruct full-resolution images from predictions.

**Key Features:**
- Loads a trained model and applies predictions to test data
- Computes fracture percentages (standard and binary)
- Merges overlapping 64×64×64 predictions into full-resolution images
- Handles overlapping tiles with proper blending
- Saves segmentation results and merged predictions

**Processing Workflow:**
1. Loads model and test data
2. Generates predictions on test blocks
3. Computes fracture statistics
4. Merges predictions using overlap-aware reconstruction
5. Saves final segmentation as 2940×2940 TIFF images

**Output:**
- Predicted segmentation blocks
- Merged full-resolution segmentation images
- Fracture percentage statistics

---

### `img_seg_3d.py`
**Purpose:** Utility functions for 3D image segmentation, block manipulation, and I/O operations.

**Key Functions:**

**Block Operations:**
- `create_3d_blocks_v1()` - Creates 3D blocks with overlap from volumetric data
- `create_3d_blocks_v2()` - Alternative block creation using view_as_blocks
- `merge_3d_blocks()` - Reconstructs images from 3D blocks
- `padding()` - Calculates padding needed to fit block size

**Image Processing:**
- `seq_sato()` - Sato ridge filter segmentation
- `seq_sato_unfiltered()` - Unfiltered Sato segmentation
- `seq_adthres()` - Adaptive threshold segmentation
- `create_ringmask()` - Creates binary mask of ring/structures
- `normalize_data()` - Normalizes data to specified bit depth

**Image Slicing & Merging:**
- `image_slicer()` - Generator for non-overlapping image patches
- `overlap_image_slicer()` - Generator for overlapping patches
- `overlap_image_slicer2()` - Alternative overlapping slicer
- `cropped_image_saver()` - Saves non-overlapping cropped patches
- `overlap_cropped_image_saver()` - Saves overlapping patches
- `merge_image()` - Reconstructs image from non-overlapping patches
- `overlap_merge_image()` - Reconstructs image from overlapping patches
- `overlap_merge_image_weka()` - WEKA-compatible overlapping merge
- `overlap_merge_image_weka2()` - Alternative WEKA merge

**I/O Operations:**
- `saveImages()` - Saves 2D images as TIFF
- `save3dImages()` - Saves 3D volumes as multi-frame TIFF
- `saveCheckImages()` - Saves binary images (1→255 scaling)
- `readingfiles()` - Lists and sorts files in directory
- `changer16to8()` - Converts 16-bit to 8-bit data

**Other:**
- `comparison()` - Visualizes segmentation results with label overlays
- `rangenbit_changer()` - Batch converts image bit depth

---

### `unet_3d_basic.py`
**Purpose:** Defines a basic 3D U-Net architecture (v3).

**Architecture:**
- 4 encoding levels with Conv3D layers
- Symmetric decoding with skip connections
- Input shape: (64, 64, 64, 1)
- Output: Single-channel binary segmentation
- Activation: ReLU for hidden layers, Sigmoid for output
- Max pooling with 2×2×2 kernels

**Customization:**
- `start_neurons` - Base number of filters (typically 32)

---

### `unet_3d_flexible.py`
**Purpose:** Defines a customizable 3D U-Net architecture with tunable parameters.

**Key Parameters:**
- `img_shape` - Input shape (default: 64×64×64×1)
- `num_class` - Output channels (typically 1)
- `start_neurons` - Base filter count
- `k_s` - Convolutional kernel size (default: 3)
- `p_s` - Pooling size (default: 2)

**Advantages over basic version:**
- Flexible kernel and pooling sizes
- Named layers for better interpretability
- 5 encoding-decoding levels
- Better fine-tuning control through named layers

**Architecture Features:**
- Batch normalization on input
- Skip connections at each level
- Binary cross-entropy loss suitable for segmentation
- Sigmoid activation for binary outputs

---

### `unet_2d_basic.py`
**Purpose:** Defines a basic 2D U-Net architecture for reference/legacy support.

**Architecture:**
- 4 encoding levels with Conv2D layers
- 2D pooling (2×2)
- Input shape: (400, 400, 1)
- Output: Single-channel binary segmentation
- Similar structure to 3D version but for 2D images

---

## Workflow Overview

### Training Pipeline
1. **Preprocess** (`preprocessing.py`): Convert raw stacks to 64×64×64 blocks
2. **Train** (`cube_model_training.py`): Train U-Net with specified parameters
3. **Monitor**: Track loss and metrics in generated plots and CSV files

### Inference Pipeline
1. **Prepare test data**: Organize test images in expected structure
2. **Predict** (`postprocess.py`): Generate predictions on test blocks
3. **Merge**: Reconstruct full-resolution images from overlapping predictions
4. **Analyze**: Compute fracture percentages and save results

## Dependencies
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scipy
- tifffile
- tqdm

## Authors
- Nicolas Hauf (nicolashauf@gmail.com)
- Dongwon Lee (dongwon.lee@mechbau.uni-stuttgart.de)

## Notes
- All scripts support GPU acceleration through TensorFlow
- Data is normalized to 8-bit unsigned integers for consistency
- Default image block size is 64×64×64 voxels
- Models are saved in Keras format (`.keras`)
