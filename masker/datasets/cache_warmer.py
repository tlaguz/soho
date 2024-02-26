import threading
import time
from queue import Queue

from tqdm import tqdm

from masker.utils import get_paths


class CacheWarmer():
    def __init__(self, create_dataset):
        self.create_dataset = create_dataset
        self.paths = get_paths()

    def warm(self):
        worker_count = 2
        workers = []

        class DatasetWorker(threading.Thread):
            def __init__(self, queue, create_dataset, *args, **kwargs):
                self.queue = queue
                self.create_dataset = create_dataset
                super().__init__(*args, **kwargs)

            def run(self):
                self.dataset = self.create_dataset()
                while True:
                    idx = self.queue.get()
                    if idx is None:  # None in queue is a signal to stop processing
                        break
                    item = self.dataset.__getitem__(idx)
                    self.queue.task_done()  # Signal that the task is done

        total_items = len(self.create_dataset())

        # Create a queue and add tasks (items to process)
        task_queue = Queue()
        for idx in range(total_items):
            task_queue.put(idx)

        for _ in range(worker_count):
            worker = DatasetWorker(task_queue, self.create_dataset)
            worker.start()
            workers.append(worker)

        for item in range(total_items):
            task_queue.put(item)

        with tqdm(total=total_items) as pbar:
            last_remaining = task_queue.qsize()
            while not task_queue.empty():
                remaining = task_queue.qsize()
                done = last_remaining - remaining
                pbar.update(done)

                last_remaining = remaining
                time.sleep(1)

        for _ in range(worker_count):
            task_queue.put(None)  # stopping signal for workers

        for worker in workers:
            worker.join()