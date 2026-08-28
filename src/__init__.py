"""
ViEmoText - Vietnamese Emotion Text Classification
Source code package for the project.
"""

__version__ = "2.0"
__author__ = "Nhóm Gà làm khoa học"

from . import data
from . import models
from . import losses
from . import utils

__all__ = ['data', 'models', 'losses', 'utils']
