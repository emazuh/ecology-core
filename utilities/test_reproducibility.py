from utilities.reproducibility import set_global_seed
import torch

def test_seed():
    set_global_seed(0)
    a = torch.rand(1)
    set_global_seed(0)
    b = torch.rand(1)
    assert torch.allclose(a, b)

if __name__ == '__main__':
    test_seed()
    print("Test reproducibility seed passed")
