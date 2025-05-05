import argparse

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Text Classification Finetuning Runner")
    
    # Required arguments
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the experiment configuration file"
    )

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser.parse_args()