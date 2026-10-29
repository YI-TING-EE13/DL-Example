"""Keep small CPU checks predictable on development and CI machines."""

import torch

torch.set_num_threads(2)
