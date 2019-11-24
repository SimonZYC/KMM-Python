'''
@Author: your name
@Date: 2019-11-24 15:28:25
@LastEditTime: 2019-11-24 15:35:40
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /KMM-Python/bin/test_kmm.py
'''
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOTDIR / 'kmm'))

from kmm import kmm

