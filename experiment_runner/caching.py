import re
from abc import ABCMeta, abstractmethod
import os

import dill
import numpy as np


class CacherABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def iteritems(self):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __contains__(self, item):
        pass


class DictionaryCacher(CacherABC):
    def __getitem__(self, key):
        if key in self:
            return self.items_dict[key]
        raise KeyError()

    def __init__(self, flush_every_n=1):
        self.flush_every_n = flush_every_n
        self.items_dict = self.load()
        self.counter = 0

    def load(self):
        pass

    def save(self):
        pass

    def __setitem__(self, key, value):
        self.items_dict[key] = value
        self.counter += 1
        if self.counter >= self.flush_every_n:
            self.save()
            self.counter = 0

    def iteritems(self):
        return self.items_dict.iteritems()

    def __len__(self):
        return len(self.items_dict)

    def keys(self):
        return self.items_dict.keys()

    def values(self):
        return self.items_dict.values()

    def __contains__(self, item):
        return item in self.items_dict


class SingleFileCacher(DictionaryCacher):
    def __init__(self, filename, flush_every_n=1):
        self.filename = filename
        super(SingleFileCacher, self).__init__(flush_every_n=flush_every_n)

    def load(self):
        try:
            with open(self.filename, 'rb') as cache_file:
                return dill.load(cache_file)
        except:
            # print('failed to load cache from %s' % self.cache_filename)
            return {}

    def save(self):
        with open(self.filename, 'wb') as cache_file:
            dill.dump(self.items_dict, cache_file, -1)


class MultipleFilesCacher(DictionaryCacher):
    def __init__(self, cache_dir, flush_every_n=1,
                 file_match_regex=r'([0-9]+)\.pkl',
                 file_name_source=lambda key: '%d.pkl' % key):
        self.cache_dir = cache_dir
        self.file_match_regex = file_match_regex
        self.file_name_source = file_name_source
        if not self.cache_dir.endswith('/'):
            self.cache_dir += '/'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        super(MultipleFilesCacher, self).__init__(flush_every_n=flush_every_n)
        self.saved_ids = set(self.items_dict.keys())

    def load(self):
        items = {}

        for filename in os.listdir(self.cache_dir):
            match = re.match(self.file_match_regex, filename)
            if match:
                try:
                    filename = '%s%s' % (self.cache_dir, filename)
                    key = int(match.groups()[0])
                    with open(filename, 'rb') as cache_file:
                        data = dill.load(cache_file)
                    items[key] = data

                except:
                    continue
        return items

    def save(self):
        for key, data in self.items_dict.iteritems():
            if key in self.saved_ids:
                continue
            filename = '%s%s' % (self.cache_dir, self.file_name_source(key))
            with open(filename, 'wb') as cache_file:
                dill.dump(data, cache_file, -1)
            self.saved_ids.add(key)


class RemoteMultipleFilesCacher(MultipleFilesCacher):
    def __init__(self, cache_dir, flush_every_n=1, file_name_source=lambda key: '%d.pkl' % key):
        super(RemoteMultipleFilesCacher, self).__init__(cache_dir, flush_every_n=flush_every_n,
                                                        file_match_regex='',
                                                        file_name_source=file_name_source)

    def load(self):
        return {}

    def save(self):
        super(RemoteMultipleFilesCacher, self).save()
        for id in self.saved_ids:
            del self.items_dict[id]
