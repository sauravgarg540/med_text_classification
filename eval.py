from ml.utils.log_utils import logger
from ml.utils.parse_args import parse_args
from config import cfg, update_config
from ml.engine import Engine


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    try:
        # Run the experiment
        runner = Engine(cfg)
        runner.evaluate()

    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise e
