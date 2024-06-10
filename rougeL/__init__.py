# Code by zzjchen
# This code includes functions for evaluation
"""
A modified ROUGE-L package which enables setting 'beta' for calculation.
Main structure & code borrowed from
    https://github.com/pltrdy/rouge/tree/master/rouge
"""

from __future__ import absolute_import
from rougeL.rouge import FilesRouge, Rouge

