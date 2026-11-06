"""First-call serialization preserves numerical inputs and caller runtime state."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import random
import threading

import numpy as np
import pytest
import torch

from milplus import numerics


def _reset(monkeypatch):
    monkeypatch.setattr(numerics, '_ready_pid', None)
    monkeypatch.setattr(numerics, '_initialization_lock', threading.Lock())


def test_serial_scalar_initialization_preserves_rng_threads_flags_and_runs_once(monkeypatch):
    _reset(monkeypatch)
    random.seed(119);np.random.seed(119);torch.manual_seed(119)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
        torch.get_num_threads(),torch.get_num_interop_threads(),torch.is_grad_enabled(),
        torch.get_default_dtype(),torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),torch.cuda.is_initialized())
    original=torch.sqrt;calls=[]
    def observed(value,*args,**kwargs):
        calls.append((value.dtype,str(value.device),value.numel(),float(value)))
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'sqrt',observed)
    with torch.no_grad():
        numerics.initialize_cpu_vector_math()
        assert not torch.is_grad_enabled()
    numerics.initialize_cpu_vector_math()
    expected=[(torch.float32,'cpu',1,4.),(torch.float64,'cpu',1,4.)] if torch.backends.mkl.is_available() else []
    assert calls==expected
    after=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
        torch.get_num_threads(),torch.get_num_interop_threads(),torch.is_grad_enabled(),
        torch.get_default_dtype(),torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),torch.cuda.is_initialized())
    assert before[0]==after[0] and before[1][0]==after[1][0]
    assert np.array_equal(before[1][1],after[1][1]) and before[1][2:]==after[1][2:]
    assert torch.equal(before[2],after[2]) and before[3:]==after[3:]


def test_concurrent_requests_serialize_the_first_vector_math_dispatch(monkeypatch):
    _reset(monkeypatch)
    barrier=threading.Barrier(6);entered=threading.Event();release=threading.Event()
    original=torch.sqrt;calls=[]
    def observed(value,*args,**kwargs):
        calls.append((threading.get_ident(),value.dtype))
        if len(calls)==1:
            entered.set()
            assert release.wait(15),'The controlled initializer was not released'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'sqrt',observed)
    def worker():
        barrier.wait(timeout=15)
        numerics.initialize_cpu_vector_math()
        return numerics._ready_pid
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker) for _ in range(6)]
        if torch.backends.mkl.is_available():
            assert entered.wait(15)
            assert len(calls)==1 and numerics._ready_pid is None
        release.set()
        assert [f.result(timeout=15) for f in futures]==[os.getpid()]*6
    if torch.backends.mkl.is_available():
        assert len(calls)==2 and calls[0][0]==calls[1][0]
        assert [v[1] for v in calls]==[torch.float32,torch.float64]
    else:assert calls==[]


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),reason='Fork lifecycle applies to POSIX workers')
def test_forked_worker_resets_a_parent_held_initialization_lock(monkeypatch):
    _reset(monkeypatch)
    numerics.initialize_cpu_vector_math()
    context=multiprocessing.get_context('fork')
    receive,send=context.Pipe(duplex=False)
    parent_pid=os.getpid()
    def child():
        initial=numerics._ready_pid
        numerics.initialize_cpu_vector_math()
        send.send((initial,numerics._ready_pid,os.getpid()))
        send.close()
    numerics._initialization_lock.acquire()
    process=context.Process(target=child)
    try:
        process.start()
    finally:numerics._initialization_lock.release()
    try:
        assert receive.poll(15),'Child inherited a locked initialization mutex'
        initial,ready,actual=receive.recv()
        process.join(timeout=15)
        assert process.exitcode==0 and initial is None and ready==actual and actual!=parent_pid
        assert numerics._ready_pid==parent_pid
    finally:
        if process.is_alive():process.terminate();process.join(timeout=5)
        receive.close();send.close()


def test_public_seeding_initializes_vector_math_before_model_rng(monkeypatch):
    from milplus import engine
    _reset(monkeypatch)
    original_initialize=engine.initialize_cpu_vector_math;original_seed=torch.manual_seed;events=[]
    def initialize():
        original_initialize();events.append('initialized')
        assert numerics._ready_pid==os.getpid()
    def seed(value):
        assert events==['initialized']
        events.append('seeded');return original_seed(value)
    monkeypatch.setattr(engine,'initialize_cpu_vector_math',initialize)
    monkeypatch.setattr(torch,'manual_seed',seed)
    engine.seed_everything(119)
    assert events==['initialized','seeded']
