#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_text_sim
----------------------------------

Tests for `text_sim` module.
"""


import sys
import unittest

from text_sim import text_sim



class TestTextSim(unittest.TestCase):

    def setUp(self):
        self.sim = text_sim.TextSim(ngrams=2)
        self.corpus = 'abc abc abc def'
        self.sim.fit_model(self.corpus)

    def test_00_functional(self):
        self.assertEqual(self.sim.num_tokens, len(self.corpus) - 1)
        print(self.sim.token_counts)
        s1 = self.sim.dice_coef('ab de', 'ab')
        s2 = self.sim.dice_coef('de ab', 'ab')
        self.assertLess(abs(s1 - s2), .05)
        s3 = self.sim.dice_coef('de ab', 'de')
        self.assertGreater(s3, s2)

        raise Exception()



class TestIndexedTextSim(unittest.TestCase):

    def setUp(self):
        self.sim = text_sim.IndexedTextSim(ngrams=2)
        self.corpus = list(enumerate('abcd abce abc def'.split()))
        self.sim.fit_model(self.corpus)

    def test_00_functional(self):
        s1 = self.sim.most_similar('abc')
        print(s1)
        s1 = self.sim.most_similar('ab')
        print(s1)
        s1 = self.sim.most_similar('defgh')
        print(s1)
        s1 = self.sim.most_similar('cde')
        print(s1)

        raise Exception()

if __name__ == '__main__':
    sys.exit(unittest.main())
