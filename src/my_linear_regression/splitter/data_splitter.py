#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Built-in imports.
import os
from copy import deepcopy

# Third-party imports.
import pandas
import numpy


# Module constants for testing.
LOCAL_USER = os.getlogin()
DATA_DIR = f'C:\\Users\\{LOCAL_USER}\\source\\repos\\LinearRegressionPython\\data\\iris.csv'


class DataSplitter():
    """
    A class that can be used to separate data into groups for training
    and testing machine learning models. \n
    Features:
        _get_training_split
        _get_testing_split
        get_feature_count
        get_sample_count
        get_training_samples
        get_testing_samples
        get_class_space
        reduce_class_space
        set_int_class_space
    """

    def __init__(
            self,
            **kwargs,):
        """
        Initializes an object that can open file objects and split the
        dataset into groups of training and testing.
        """
        self.file_path = kwargs.get('file_path')
        self.split = kwargs.get('split')
        self.cls_space_size = kwargs.get('cls_space_size')
        self.cls_field = kwargs.get('cls_field')
        self.y_seed = kwargs.get('y_seed')
        self.y_interval = kwargs.get('y_interval')
        self.prepare = kwargs.get('prepare')

        try:
            self.data = pandas.read_csv(self.file_path)
        except BaseException as e:
            print(e)
        
        if self.prepare: # review this.
            self.reduce_class_space(
                size=self.cls_space_size,
                field=self.cls_field)
            self.set_int_class_space(
                cls_field=self.cls_field,
                new_field='class_int_rep',
                y_seed=self.y_seed,
                y_interval=self.y_interval)


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

    def get_feature_count(self) -> int:
        """
        Returns the number of features in this objects data frame. \n
        Returns:
            int
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> assert ds.get_feature_count() == 5
        """
        return len(self.data.columns)

    def get_sample_count(self) -> int:
        """
        Returns the number of samples in this objects data frame. \n
        Returns:
            int
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> assert ds.get_sample_count() == 150
        """
        return len(self.data)

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

    def get_training_samples_array(
            self) -> numpy.array:
        """
        Returns the training portion of the data set as numpy array
        without class information \n
        Returns:
            pandas.DataFrame
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> ts = ds.get_training_samples_array()
            >>> assert ts.iloc[-1][0] == 5.2
        """
        return numpy.array(self.get_training_samples())

    def get_testing_samples(
            self) -> pandas.DataFrame:
        """
        Returns the testing portion of the data set. \n
        Returns:
            pandas.DataFrame
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> ts = ds.get_testing_samples()
            >>> assert ts.iloc[0][0] == 5
        """
        return self.data.iloc[self._get_training_split(): ]

    def get_testing_samples_array(
            self) -> numpy.array:
        """
        Returns the testing portion of the data set as numpy array
        without class information \n
        Returns:
            pandas.DataFrame
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> ts = ds.get_testing_samples_array()
            >>> assert ts.iloc[-1][0] == 5.2
        """
        return numpy.array(self.get_testing_samples())

    def get_class_space(
            self,
            field: str) -> list:
        """
        Returns the distinct number of class included in data. \n
        Returns:
            int
        Doctest:
            >>> ds = DataSplitter(DATA_DIR)
            >>> e = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            >>> assert sorted(ds.get_class_space('species')) == e
        """
        return list(set([
            x
            for x in self.data[field]]))

    def reduce_class_space(
            self,
            size: int,
            field: str,):
        """
        Reduces the class space for the dataset to n classes. \n
        """
        n = len(self.get_class_space(field)) - size
        if n < 0:
            raise ValueError('Class space would be reduced to 0 or'
                            'fewer classes.')

        classes = self.get_class_space(field)[0: n]
        indexes = self.data[self.data[field].isin(classes)].index
        self.data.drop(indexes, inplace=True)

    def set_int_class_space(
            self,
            cls_field: str,
            new_field: str,
            y_seed: int,
            y_interval: int,):
        """
        If the integer representation of class field does not
        exist, create it. Set the class space tags to integer
        values from the y_seed increasing by y_interval untill
        each class has an int representation.
        """
        if not new_field in self.data.columns:
            self.data[new_field] = None
        
        classes = self.get_class_space(cls_field)
        for i, c in enumerate(classes):
            self.data.loc[
                self.data[cls_field]==classes[i], 
                [new_field]] = y_seed
            y_seed += y_interval

    def set_field(
            self,
            key: str,
            value):
        """
        Add a field onto the self.data data frame
        """
        self.data[key] = value

if __name__ == "__main__":
    import doctest
    doctest.testmod()
