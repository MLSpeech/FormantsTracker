import hydra
import torch
from argparse import Namespace
from solver import Solver

@hydra.main(version_base=None,config_path='conf', config_name='config')
def main(cfg):
    cfg = Namespace(**dict(cfg))
    cfg.device = torch.device("cuda" if (torch.cuda.is_available() and cfg.is_cuda) else "cpu")
    solver = Solver(cfg)
    solver.test()
    print("!~Done~!")


if __name__ == "__main__":
    main()
