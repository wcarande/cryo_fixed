"""
Example of a unit test for band ratio calculation.

Created on: July 3, 2021

@author Wendy Carande
"""
import os
import unittest

from . import band_ratio


class BandRatioTestCase(unittest.TestCase):
    def test_band_ratio_tx300(self):
        data_dir = '../../data'
        file_list = os.listdir(data_dir)
        find_band_ratio = band_ratio.FindBandRatio(data_dir, file_list)

        file = '2002tx300.txt'
        ratio_actual = find_band_ratio.find_ratio(file, plotting=False)

        # The value 0.0898 comes from the IDL version of this script written in
        # 2008 for the same object data (2002 tx300).
        self.assertAlmostEqual(0.0898, ratio_actual, 2)


if __name__ == '__main__':
    unittest.main()
