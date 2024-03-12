import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig)-> None:
    pl.seed_everything(cfg.seed)
    print(cfg)


if __name__ == "__main__":
    main()