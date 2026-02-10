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
