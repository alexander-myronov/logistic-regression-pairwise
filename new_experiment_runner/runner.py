from __future__ import division, print_function

import sys
import traceback

from new_experiment_runner.cacher import CSVCacher

__author__ = 'myronov'

# coding: utf-8

import itertools

from tqdm import tqdm as tqdm


class Runner(object):
    def __init__(self, task, task_generator, mapper=itertools.imap,
                 cacher=CSVCacher(filename=None)):
        self.task = task
        self.task_generator = task_generator
        self.mapper = mapper
        self.cacher = cacher

    @staticmethod
    def map_f(task_context_kwds):
        task, context, kwds = task_context_kwds
        try:
            return context, task(context, **kwds)
        except Exception as e:
            e_t, e_v, e_tb = sys.exc_info()
            e_tb = traceback.format_tb(e_tb)
            return context, (e_t, e_v, e_tb)

    def run(self):

        tasks = map(lambda (context, kwds): (self.task, context, kwds), self.task_generator)
        tq = tqdm(total=len(tasks))
        for context, result in self.mapper(Runner.map_f, tasks):

            if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], Exception):
                print(result[0])
                print(result[1])
                print('\n'.join(result[2]))
                continue

            if not isinstance(result, dict):
                result = {'result': result}

            self.cacher.set(context, result)
            self.cacher.save()
            tq.update()
        tq.close()


if __name__ == '__main__':

    import multiprocess as mp

    mp.freeze_support()


    def task_generator():
        context = {}
        for i in xrange(100):
            context['i'] = i
            yield dict(context), {'kwd1': 100500 + i}


    def task(context, **kwargs):
        return context['i'] + kwargs.pop('kwd1')


    cacher = CSVCacher(filename=None)

    runner = Runner(task, task_generator(), cacher=cacher,
                    mapper=mp.Pool(processes=2).imap_unordered)
    runner.run()
    print(runner.cacher.dataframe)
