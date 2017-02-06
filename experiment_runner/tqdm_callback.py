from tqdm import tqdm


class TqdmCallback(object):
    def __init__(self):
        self.progress_bar = None
        self.value = 0

    def __call__(self, update_progress=0, length=0, description=None):
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=length)
            self.value = 0
        self.progress_bar.update(n=update_progress)
        self.value += update_progress
        if description:
            self.progress_bar.set_description(description)
        if self.progress_bar and self.value == length:
            self.progress_bar.close()
            self.progress_bar = None
            self.value = 0
