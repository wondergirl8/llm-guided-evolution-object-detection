"""Standalone Faster R-CNN baseline entry point (no LLM-GE dependency)."""
if __package__:
    from .train_eval import cli
else:
    from train_eval import cli

if __name__ == "__main__":
    cli()
