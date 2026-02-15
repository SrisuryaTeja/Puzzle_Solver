import torch
import random
import numpy as np

from .train import train
from .inference import reconstruct_beam
from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle, reset_id_generators


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    set_seed(42)

    image_paths = [
        r"D:\srisurya\Ml_projects\FaceFirst\Puzzle_Solver\City_Scape.jpg",
        r"D:\srisurya\Ml_projects\FaceFirst\Puzzle_Solver\dock.jpg",
        r"D:\srisurya\Ml_projects\FaceFirst\Puzzle_Solver\Forrest.jpg"
    ]

    # TRAIN

    print("Starting training...")
    train(image_paths)

    

if __name__ == "__main__":
    main()
