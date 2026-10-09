"""Compatibility entry point; training and shared evaluation live in train_eval."""
try:
    from .train_eval import main, run
except ImportError:
    from train_eval import main, run

if __name__ == "__main__":
    raise SystemExit(main())
