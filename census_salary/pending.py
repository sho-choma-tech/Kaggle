import argparse
import inspect
import re 
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd 
import time 
from contextlib import contextmanager

#context managerとは，with 文の実行時にランタイムコンテキストを定義するオブジェクト

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force', '--f', action='store_true', help='Overwrite existing files'
    )

    return parser.parse_args()

def get_features(namespace, overwrite):
    