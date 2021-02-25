import json
import os
import unittest
from functools import lru_cache

import numpy as np

from karel_for_synthesis.deltas import compute_deltas, run_deltas


@lru_cache(None)
def get_data():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs.json")
    ) as f:
        sparse = json.load(f)
    pairs = np.zeros(sparse["shape"])
    pairs[tuple(sparse["indices"])] = 1
    return pairs


class TestDeltas(unittest.TestCase):
    def test_basic_deltas(self):
        for input, output in get_data():
            deltas = compute_deltas(input, output)
            new_output = run_deltas(deltas, input)
            self.assertEqual(np.array(np.where((new_output != output))).T.tolist(), [])
