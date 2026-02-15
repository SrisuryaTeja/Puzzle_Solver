# Image Square Reassembly

This project asks you to unscramble an input image, that has been divided into a grid of square tiles. 
The tiles will be shuffled and potentially rotated. The task will be to develop an ML model to re-arrange 
the tiles to their original state. 

To facilitate the task, the tile in the top left corner will not be shifted nor rotated. 

Here is the dock.jpg image scrambled.

<img width="789" height="790" alt="image" src="https://github.com/user-attachments/assets/c28dfce4-cfb4-45c6-a778-bc029034e5ff" />

Here is the image re-assembled. 

<img width="789" height="790" alt="image" src="https://github.com/user-attachments/assets/6b18c51b-b0f8-458f-92b9-f656a62077e1" />

## Overview

Much of this is demonstrated in the Overview_of_Puzzles.ipynb notebook.

- Load an image
- Split the image into equal-sized square pieces
- Randomly shuffle the squares
- Please use python 3.11.7 and the create_virtual_env.py to generate a virtual env. with the required modules.

Your task: 
- Train a model to predict which tile shares a boundary with another tile
- Use your model to reconstruct the original image from the model’s output 
- You cannot use any images besides the 3 provided for your training data

## Goal

- Develop a data strategy to train the model
- Design and train a model to  identify which tiles should be next to each other
- Use your model to create a method to recreate the original image from the shuffled tile

## When you are done MAKE A PR to this repo!


## My Solution

I have trained a model to:

- Predict whether two puzzle tiles share a boundary

- Use those predictions to reconstruct a shuffled image

- Train using only the three provided images (as required)

The core idea is to treat puzzle solving as a boundary compatibility learning problem, rather than a full image classification problem.

Instead of reasoning about entire tiles, the model learns whether two tile edges match.


## Approach

The solution consists of two main stages:

- Learning Edge Compatibility (Training Phase)

We train a Siamese neural network to learn whether two tile edges belong next to each other.

Instead of using full tiles, we extract thin edge strips and learn a similarity function between them.

- Image Reconstruction (Inference Phase)

    Using the learned compatibility score:

    - We reconstruct the puzzle using Beam Search

    - At each grid position:

        - Try every unused tile

        - Try all 4 rotations

        - Score compatibility with placed neighbors

    - Keep top-k candidates (beam_width)

    - Render the best scoring arrangement


## 🏗 Data Strategy (Core Design)

One of the most important challenges was ensuring rotation-invariant supervision.

Rotation-Safe Label Generation

  - Split image into grid without rotation

  - Generate ground-truth adjacency using internal edge IDs

  - Apply random rotations after labels are generated

      - This guarantees:

          - Correct supervision

          - Rotation robustness

          - No label corruption

##  Why Edge Strips Instead of Full Tiles?

Instead of feeding full tiles into the model, we extract **narrow boundary strips**:

- Right ↔ Left
- Bottom ↔ Top

Each strip is:
- Normalized to a canonical horizontal orientation
- Flipped (for one side) to ensure proper boundary alignment

### Benefits

-  Reduces input dimensionality  
-  Focuses the model strictly on boundary information  
-  Removes unnecessary interior visual noise  
-  Makes similarity learning significantly easier  

---

##  Model Architecture

###  Edge Encoder

A lightweight CNN encoder composed of:

- Conv → BatchNorm → ReLU (3 blocks)
- Adaptive Average Pooling
- Fully connected projection layer
- L2 normalization of embeddings

This produces a fixed-dimensional embedding vector for each edge strip.

---

###  Siamese Compatibility Network

- Shared encoder for both input strips (Siamese architecture)
- Similarity computed using cosine similarity
- Trained using `CosineEmbeddingLoss`

This ensures:

-  A geometrically meaningful embedding space  
-  Direct compatibility scoring during inference  
-  Stable metric learning behavior  

---

##  Reconstruction Algorithm

We use **Beam Search** to assemble the puzzle.

### Steps

1. Fix an anchor tile at position `(0, 0)`
2. For each grid position:
   - Try every unused tile
   - Try all 4 possible rotations
   - Score compatibility against:
     - Left neighbor (if exists)
     - Top neighbor (if exists)
3. Keep the top-k scoring states (beam width)
4. Continue until the grid is completely filled


## Training Details

###  Images Used

The model was trained **exclusively** on the three provided images:

- `City_Scape.jpg`
- `dock.jpg`
- `Forrest.jpg`

No external images or pretrained weights were used.

---

###  Grid Sizes

Training was performed using multiple puzzle resolutions:

- 3 × 3  
- 4 × 4  
- 5 × 5  
- 6 × 6  

This improves generalization across different puzzle complexities.

---

###  Optimization Setup

- **Loss Function:** `CosineEmbeddingLoss`
- **Optimizer:** Adam
- **Epochs:** 40
- **Batch Size:** 32

---

### Model Checkpoint

Best performing model is saved as `Cosine_Embedding_Loss_model.pth`


