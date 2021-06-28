"""
Finds the ratio of the 1.65 micron feature area to the entire spectrum area.

Created on: June 28, 2021

@author Wendy Carande
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


class FindBandRatio:
    """ Class to find the band ratio.

    """
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.file_list = file_list

        # Set the minimum and maximum micron values for the whole spectrum range
        self.min_micron = 1.4
        self.max_micron = 1.69

        # Set the minimum micron value for the 1.65 micron feature.
        # This is typically around 1.626.
        self.min_micron_165 = 1.626

        # Set a few values for dataframe columns
        self.wl_string = 'wavelength'
        self.refl_string = 'reflectance'
        self.line_string = 'line'

        # Set plot colors
        self.color_full_area = 'purple'
        self.color_165_area = 'green'

    def find_slope_intercept(self, df):
        """ Find the slope and intercept  between a line drawn from the first
            point to the last point of a sorted dataframe of wavelengths and
            reflectances. Uses scipy stats linregress.

            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

            Parameters:
            df (pandas.dataframe): dataframe with columns wavelength and
                reflectance

            Returns:
            float: slope of line between first and last points
            float: intercept

        """
        wavelength_start = df[self.wl_string].iloc[0]
        wavelength_stop = df[self.wl_string].iloc[-1]
        reflectance_start = df[self.refl_string].iloc[0]
        reflectance_stop = df[self.refl_string].iloc[-1]

        slope, intercept, _, _, _ = linregress([wavelength_start, wavelength_stop],
                                               [reflectance_start, reflectance_stop])
        return slope, intercept

    def plot_spectrum(self, df, color):
        """ Find the slope and intercept  between a line drawn from the first
            point to the last point of a sorted dataframe of wavelengths and
            reflectances.

            Parameters:
            df (pandas.core.frame.DataFrame): dataframe with columns wavelength
                and reflectance
            color (str): string with a valid matplotlib color value for plotting

            Returns:
            numpy.float64: slope of line between first and last points
            numpy.float64: intercept

        """
        plt.scatter(df[self.wl_string], df[self.refl_string], color=color)
        plt.plot(df[self.wl_string], df[self.line_string], color=color)

        # Plot the fill between the line and the spectrum points. alpha sets a
        # transparency.
        plt.fill(np.append(df[self.wl_string], df[self.wl_string][::-1]),
                 np.append(df[self.line_string], df[self.refl_string][::-1]),
                 color, alpha=0.25)

    def find_area(self, df):
        """ Find the difference in the area under the curve for a line between
            the start and end points and the reflectance values for a sorted
            dataframe. Uses the trapz function from numpy to find the integrated
            areas.

            See: https://numpy.org/doc/stable/reference/generated/numpy.trapz.html

            Parameters:
            df (pandas.core.frame.DataFrame): dataframe with columns wavelength
                and reflectance

            Returns:
            numpy.float64: difference in area

        """
        area_below_line = np.trapz(df[self.line_string],
                                   df[self.wl_string])  # Area below the line between the start and end points
        area_below_point = np.trapz(df[self.refl_string], df[self.wl_string])  # Area below the reflectance curve
        diff_area = area_below_line - area_below_point
        return diff_area

    def process_spectrum_dataframe(self, df):
        """ Given an input dataframe of raw spectrum data, create a dataframe
            with filtered and sorted values and add a column with points on a
            line between the first and last values of the spectrum.

            Parameters:
            df (pandas.core.frame.DataFrame): dataframe with columns wavelength
                and reflectance (raw values)

            Returns:
            pandas.core.frame.DataFrame: dataframe with filtered and sorted
                spectrum values and a line column

        """
        # Filter the data between the min and max micron values
        df_spectrum = df[(df[self.wl_string] >= self.min_micron) &
                         (df[self.wl_string] < self.max_micron)]

        # Sort the dataframe inplace by wavelength then reflectance
        df_spectrum.sort_values(by=[self.wl_string, self.refl_string],
                                ascending=True, inplace=True)

        # Reset the dataframe indices
        df_spectrum.reset_index(inplace=True)

        # Uncomment the next line to plot just the spectrum
        # df_spectrum.plot(x=self.wl_string, y=self.refl_string, kind='scatter')

        # Find the slope and intercept; create a new column in the dataframe with the line values
        slope, intercept = self.find_slope_intercept(df_spectrum)
        df_spectrum[self.line_string] = slope * df_spectrum[self.wl_string] + intercept  # y = mx + b

        # Uncomment next line to plot the spectrum and the line between first and last points
        # df_spectrum.plot(x=self.wl_string, y=[self.refl_string,self.line_string])

        return df_spectrum

    def process_165_feature_dataframe(self, df):
        """ Given an input dataframe of processed spectrum data, create a
            dataframe filtered to the 1.65 micron feature values and add a
            column with points on a line between the first and last values of
            the 1.65 micron feature spectrum.

            Parameters:
            df (pandas.core.frame.DataFrame): dataframe with columns wavelength
                and reflectance and line values

            Returns:
            pandas.core.frame.DataFrame: dataframe with 1.65 micron feature
                values

        """
        # Find the index where the 1.65 micron spectral feature starts.
        ind_list = df[df[self.wl_string] == self.min_micron_165].index

        # Take the last index from the list
        if len(ind_list > 0):
            ind_max = ind_list[-1]

        # If there are no values in the index list, look for the index of the
        # last value before the stated start value.
        else:
            ind_list = df[df[self.wl_string] < self.min_micron_165].index
            ind_max = ind_list[-1]

        # Define a new dataframe just with the data for the 1.65 micron feature
        # and reset indices
        df_spectrum_165 = df[df.index >= ind_max].drop(self.line_string, axis=1)
        df_spectrum_165.reset_index(inplace=True)

        # Find the slope and index for the line between the first and last
        # points of the 1.65 micron feature. Create a new column in the
        # dataframe to hold the line values.
        slope_165, intercept_165 = self.find_slope_intercept(df_spectrum_165)
        df_spectrum_165[self.line_string] = slope_165 * df_spectrum_165[self.wl_string] + intercept_165

        return df_spectrum_165

    def find_ratio(self, file):
        """ Find the ratio of the area difference for the 1.65 micron spectral
            line to the area difference for the entire spectrum.

            Parameters:
            file (str): string with file name

            Returns:

        """
        # Object name
        obj = file.replace('.txt', '')  # replace .txt with blank space

        # Create a dataframe from csv data.
        #  sep (delimiter) is '/t' for tab, must set header to None
        df_spectrum_orig = pd.read_csv(os.path.join(self.data_dir, file),
                                       sep='\t', header=None,
                                       names=[self.wl_string, self.refl_string])

        # Process the spectrum dataframe
        df_spectrum = self.process_spectrum_dataframe(df_spectrum_orig)

        # Process the 1.65 micron feature dataframe
        df_spectrum_165 = self.process_165_feature_dataframe(df_spectrum)

        # Find the ratio of area difference of the complete spectrum to the
        # area difference of the 1.65 micron feature
        area_total = self.find_area(df_spectrum)
        area_165 = self.find_area(df_spectrum_165)
        ratio = area_165 / area_total

        # Plot the full spectrum and 1.65 micron feature areas
        plt.figure()
        self.plot_spectrum(df_spectrum, self.color_full_area)  # Full spectrum
        self.plot_spectrum(df_spectrum_165, self.color_165_area)  # 1.65 micron feature
        plt.ylabel('Relative Reflectance')
        plt.xlabel('Wavelength [microns]')
        plt.title(f'{obj}: ratio={ratio:.3}')  # The .3 means print 3 sig figs
        plt.show()
