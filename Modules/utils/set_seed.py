
def set_seed(seed: int = 42):
    """
    Set a random state seed for both torch and cuda
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
