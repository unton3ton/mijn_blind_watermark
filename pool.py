import sys, multiprocessing, warnings

if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')

class CommonPool(object):
    def map(self, func, args):
        return list(map(func, args))

class AutoPool(object):
    def __init__(self, mode, processes):
        if mode == 'multiprocessing' and sys.platform == 'win32':
            warnings.warn('multiprocessing not support in windows, turning to multithreading')
            mode = 'multithreading'
        self.mode = mode
        self.processes = processes
        if mode == 'multithreading':
            from multiprocessing.dummy import Pool as ThreadPool
            # Этот параметр устанавливает количество воркеров в пуле.
            # Если оставить это поле пустым, то по умолчанию оно будет равно количеству ядер в вашем процессоре.
            self.pool = ThreadPool(processes=processes) 
        elif mode == 'multiprocessing':
            from multiprocessing import Pool
            self.pool = Pool(processes=processes)
        else:  # common
            self.pool = CommonPool()

    def map(self, func, args):
        return self.pool.map(func, args)