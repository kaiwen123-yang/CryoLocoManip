"""Stage 0B.2A runtime/checkpoint/training-feasibility tools.

Modules here audit whether the released BorealTC CNN and Mamba code paths can
be prepared for formal training reproduction on the current host. They never
modify the pinned upstream checkout and never run full-epoch or full-fold
training.
"""
