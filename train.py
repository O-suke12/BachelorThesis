import hydra
import pytorch_lightning as pl

from tbsim.algos.factory import algo_factory
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    print(cfg)
    datamodule = datamodule_factory(cls_name=cfg.train.datamodule_class, config=cfg)
    datamodule.setup()
    model = algo_factory(config=cfg, modality_shapes=datamodule.modality_shapes)


if __name__ == "__main__":
    main()
    main()
