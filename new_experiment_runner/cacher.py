import itertools
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

__author__ = 'myronov'


class CacherABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_index(self, key):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def set(self, key, value):
        pass

    @abstractmethod
    def get(self, key):
        pass


class CSVCacher(CacherABC):
    def __init__(self, filename):
        self.filename = filename
        try:
            self.dataframe = pd.read_csv(filename)
        except Exception as e:
            self.dataframe = pd.DataFrame(index=pd.RangeIndex(0))

    def __len__(self):
        return len(self.dataframe)
        pass

    def get_index(self, key):
        index = np.ones(shape=len(self.dataframe), dtype=bool)
        for k, v in key.iteritems():
            #print(k, index)
            if index.sum() == 0:
                break
            index = index & (self.dataframe[k] == v).values
        #print(index)
        return index

    def save(self):
        self.dataframe.to_csv(self.filename, index=False)

    def set(self, key, value):
        for col in key.keys() + value.keys():
            if col not in self.dataframe.columns:
                self.dataframe.loc[:, col] = pd.Series()

        current_index = dict(key)
        row_index = None
        for k, v in itertools.chain(key.iteritems(), value.iteritems()):
            current_index[k] = v
            if row_index is None:
                row_index = self.get_index(current_index)
                if row_index.sum() == 0:
                    row_index = len(self.dataframe.index)
            self.dataframe.loc[row_index, k] = v

    def get(self, key={}):
        index = self.get_index(key)
        return self.dataframe.ix[index]


if __name__ == '__main__':
    cacher = CSVCacher('data/test.csv')

    cacher.set(key={'seed': 0}, value={'value': 100})
    cacher.set(key={'seed': 0, 'value': 100}, value={'string': 'aaa'})

    # print(cacher.get({'seed': 0}))
    print(cacher.get())
