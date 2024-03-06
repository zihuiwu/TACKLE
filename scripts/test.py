import argparse
from codesign.config.config import CodesignTestConfigurator

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='MRI Codesign')
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        help="Path to config file"
    )
    parser.add_argument(
        "--data-config", "-d",
        type=str, 
        default=None,
        help="Name of the data module"
    )
    parser.add_argument(
        "--id", "-i",
        type=str, 
        default=None,
        help="Wandb ID of the run to be tested"
    )
    parser.add_argument(
        "--fix", "-f",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
    )
    args = parser.parse_args()

    # initialize exp, ckpt, model, and data_module
    cc = CodesignTestConfigurator(args)
    exp, model, data_module = cc.init_all()
    
    exp.test_run(args, model, data_module)
