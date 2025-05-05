from dotenv import load_dotenv
from ml.utils.log_utils import logger
from config import cfg, update_config
from ml.utils.parse_args import parse_args
from ml.engine import Engine

# Load environment variables
load_dotenv(".env")



def main(cfg):
    """Main execution function."""
    try:
        # Run the experiment
        runner = Engine(cfg)
        runner.finetune()

        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # initialize config from config path passed as argument
    args = parse_args()
    update_config(cfg, args)
    main(cfg)







