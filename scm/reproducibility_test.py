#!/usr/bin/python
"""Basic reproducibility test."""

import hashlib
import os
import unittest

class TestSamePickleFiles(unittest.TestCase):
    """Test that pickle files are same in jupyter and python versions"""
    results_dirnames = ['../pgm/results/jupyter', 'results/python']
    basenames = ['figure-3.p', 'figure-4.p', 'scores-and-cdfs.p']

    def test_same_pickle_files(self):  # pylint: disable=missing-docstring
        # NOTE: we assume that `python fico_figures.py` was run w/ default args

        for basename in self.basenames:
            filenames = [
                os.path.join(dirname, basename)
                for dirname in self.results_dirnames
                ]
            # NOTE: For now I check consistent outputs via checksum, but if I
            #       start saving more stuff in the output pickle files then I'll
            #       have to instead compare the objects inside the pickle files.

            cksums = []
            for filename in filenames:
                with open(filename, 'rb') as f:
                    cksums.append(hashlib.md5(f.read()).hexdigest())

            msg = 'Checksums in files {} and {} don\'t match!' \
                  .format(*filenames)
            self.assertEqual(cksums[0], cksums[1], msg)

if __name__ == '__main__':
    unittest.main()
