from abc import ABCMeta, abstractmethod
from time import sleep

import ipyparallel as ipp


class IPyParallelMappper(object):
    def __init__(self, view, sleep_interval, chunksize, verbose=False):
        self.view = view
        self.sleep_interval = sleep_interval
        self.chunksize = chunksize
        self.verbose = verbose

    def map(self, function, iterable):
        self.view.block = False
        if self.verbose:
            print('calling map')

        try:
            out = self.view.map_async(function, iterable,
                                      ordered=False,
                                      chunksize=self.chunksize)

            if self.verbose:
                print('map called')

            if not isinstance(out, list):

                client = self.view.client
                pending = set(out.msg_ids)
                while pending:
                    try:
                        self.view.wait(pending, 1e-3)
                    except ipp.TimeoutError:
                        # ignore timeout error, because that only means
                        # *some* jobs are outstanding
                        pass
                    # update ready set with those no longer outstanding:
                    ready = pending.difference(self.view.outstanding)
                    # update pending to exclude those that are finished
                    pending = pending.difference(ready)
                    while ready:
                        msg_id = ready.pop()
                        child = out._children[out.msg_ids.index(msg_id)]
                        ar = ipp.AsyncResult(client, child)
                        rlist = ar.get()
                        try:
                            for r in rlist:
                                index, gs_params = r
                                yield (index, gs_params)
                        except TypeError:
                            # flattened, not a list
                            # this could get broken by flattened data that returns iterables
                            # but most calls to map do not expose the `flatten` argument
                            pass

                    sleep(self.sleep_interval)
        except Exception as e:
            if isinstance(e, ipp.RemoteError):
                e.print_traceback()
            raise e
