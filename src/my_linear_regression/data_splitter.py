#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Built-in imports.
import os
import pandas
from copy import deepcopy


# Module constants for testing.
LOCAL_USER = os.getlogin()
DATA_DIR = f'C:\\Users\\{LOCAL_USER}\\source\\repos\\LinearRegressionPython\\data\\iris.csv'


class DataSplitter():
    """
    A class that can be used to separate data into groups for training
    and testing machine learning models.
    """

    def __init__(
            self,
            file_path: str,
            split: float = 0.4):
        """
        Initializes an object that can open file objects and split the
        dataset into groups of training and testing.
        """
        self.file_path = file_path
        self.split = split
        try:
            self.data = pandas.read_csv(self.file_path)
        except BaseException as e:
            print(e)

    @DeprecationWarning
    def _opens(
            self):
        """
        Read content of file at file path specified on initialization.
        """
        with open(self.file_path, 'r') as io:
            reader = list(csv.reader(
                                io,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_NONE))
            self.header.append(reader)
            for line in reader:
                self.data.append(line)
            io.close()

    def _get_training_split(self) -> int:
        """
        Return the ith to subscript the data for training samples. \n
        Returns:
            int
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> assert ds._get_training_split() == 60
        """
        return round(
                    len(self.data)
                    * self.split)

    def _get_testing_split(self) -> int:
        """
        Return the ith to subscript the data for testing samples. \n
        Returns:
            int
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> assert ds._get_testing_split() == 90
        """
        l = len(self.data)
        return l - round(
                    l
                    * self.split)
    
    def get_training_samples(
            self) -> pandas.DataFrame:
        """
        Returns the training portion of the data set. \n
        Returns:
            pandas.DataFrame
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> ts = ds.get_training_samples()
            >>> assert ts.iloc[-1][0] == 5.2
        """
        return self.data.iloc[0: self._get_training_split()]

    def get_testing_samples(
            self) -> pandas.DataFrame:
        """
        Returns the testing portion of the data set. \n
        Returns:
            pandas.DataFrame
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> ts = ds.get_testing_samples()
            >>> assert ts.iloc[0][0] == 5.2
        """
        return self.data.iloc[self._get_training_split() - 1: ]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
