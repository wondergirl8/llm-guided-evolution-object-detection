"""Standalone scratch Faster R-CNN seed baseline."""
try: from .train_eval import cli
except ImportError: from train_eval import cli
if __name__=="__main__": cli()
