# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:13:26 2021

@author: Sourabh
"""

import time
import numpy as np
from scipy.sparse import csr_matrix

def commmat(m,n):
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K

def vec(arr):
    return np.reshape(arr, (-1,1), order='F')

def vecinv(arr, M, N):
    return np.concatenate([arr[M*i:M*i + M, 0] for i in range(N)]).reshape((N,M)).T

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, name, toprint=True):
        self._start_time = None
        self.name = name
        self.toprint = toprint

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        if self.toprint:    
            print('Starting ' + self.name)        
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if self.toprint:
            print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()