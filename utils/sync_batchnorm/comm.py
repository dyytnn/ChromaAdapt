"""
    Description: This file implements a communication framework for synchronized operations between master and slave processes/devices, typically used in distributed pytorch training with synchronized batch normalization.
    
    Concurrency Guarantees:
        1. Thread Safety: All operations are protected by the lock
        2. Happens-Before Relationship: put() establishes happens-before get(), ensuring memory
        visibility
        3. Blocking Synchronization: get() blocks until put() is called
        4. Single Producer/Consumer: Designed for 1:1 communication only

    Both threading and asyncio provide synchronization primitives, but they serve fundamentally different concurrency models. Let me explain the key differences:

    Concurrency Models

    Threading (Preemptive Concurrency)
        - OS threads: Real system threads managed by the OS scheduler
        - Preemptive: OS can interrupt and switch threads at any time
        - Parallel execution: True parallelism on multi-core systems
        - Blocking operations: When a thread waits, it's genuinely blocked

    Asyncio (Cooperative Concurrency)
        - Single-threaded: All code runs in one OS thread
        - Cooperative: Tasks voluntarily yield control using await
        - Concurrent, not parallel: Tasks interleave execution but don't run simultaneously
        - Non-blocking: "Blocking" operations yield control to other tasks

    Synchronization Primitives Comparison

    | Feature           | threading.Lock                                 |
    asyncio.Lock                      |
    |-------------------|------------------------------------------------|---------
    --------------------------|
    | Acquisition       | with lock: (blocks thread)                     | async 
    with lock: (yields control) |
    | Blocking behavior | Thread sleeps/blocks                           | Task
    yields, event loop continues |
    | Performance       | Thread switching overhead                      | Fast
    context switching            |
    | Use case          | Protecting shared state in multi-threaded code |
    Serializing access in async code  |

    Code Examples

    Threading approach:
    import threading
    import time

    lock = threading.Lock()

    def worker(name):
        with lock:  # Thread blocks here if lock is held
            print(f"{name} acquired lock")
            time.sleep(1)  # Thread is blocked, no other work happens
        print(f"{name} released lock")

    # Creates real OS threads
    threading.Thread(target=worker, args=("Thread1",)).start()
    threading.Thread(target=worker, args=("Thread2",)).start()

    Asyncio approach:
    import asyncio

    lock = asyncio.Lock()

    async def worker(name):
        async with lock:  # Task yields control if lock is held
            print(f"{name} acquired lock")
            await asyncio.sleep(1)  # Yields control, other tasks can run
        print(f"{name} released lock")

    # Single thread, tasks cooperatively multitask
    async def main():
        await asyncio.gather(
            worker("Task1"),
            worker("Task2")
        )

    asyncio.run(main())

    When to Use Which?

    Use Threading when:

    - CPU-bound work with true parallelism needed
    - Blocking I/O that can't be made async
    - Legacy code already using threads
    - External libraries that aren't async-aware

    Use Asyncio when:

    - I/O-bound operations (network, file operations)
    - High concurrency with many simultaneous connections
    - Single-threaded performance is sufficient
    - Modern async ecosystem (aiohttp, asyncpg, etc.)
"""

import collections
import queue
import threading
from typing import Any, List, Tuple

__all__ = ["FutureResult", "SlavePipe", "SyncMaster"]

class FutureResult(object):
    """
        A thread-safe comunication mechanism that acts as one to one pipe. It uses threading locks and conditions to safely pass results between threads.
        
        FutureResult Class - Deep Dive Theoretical Foundation:
        FutureResult implements the Future/Promise pattern - a fundamental concurrency design pattern
        where:
            - A Future represents a value that will be available at some point in the future
            - It allows asynchronous operations without blocking the caller immediately
            - Provides thread-safe coordination between producer and consumer threads        
    """
    def __init__(self) -> None:
        self._result = None # store the actual result
        self._lock = threading.Lock() # mutual exclusion primitive
        self._cond = threading.Condition(self._lock) # Synchronization primitive
    
    """
        Lock: Ensure only one thread can access _result at a time (mutual exclusion)
        Condition Variable: Allows threads to wait for a specific condition (resut availability) and be notified when the condition is changed
        The condition is bound to the lock, creating an atomic wait-notify mechanism
    """
    
    def put(self, result):
        """
            Theory:
                - Producer operation: The thread that has computed a result calls this.
                _ Asserttion: Ensures single-use semantics - prevents overwriting unfetched results
                - Notify: Wakes up exactly one thread waiting on this condition (if any)
                - Context manager (with): Guarantees lock is released even if exception occurs.
        """
        with self._lock: # Acqurire exclusive access
            assert self._result is None, "cannot send value twice"
            self._result = result # store the result
            self._cond.notify() # wake up waiting thread
    
    def get(self):
        """
            Theory:
                - Consumer operation: The thread that needs the result calls this.
                - Blocking behavior: If no result exists, thread enters waiting state.
                - Spurious wakeup protection: The if check ensures we actually have a result.
                - Single-use consumption: Result is cleared after retrieval, enabling reuse.
        """
        with self._lock:
            if self._result is None:
                self._cond.wait()
            
            res = self._result
            self._result = None
            return res
        
_MasterRegistry = collections.namedtuple("MasterRegistry", ["result"])
_SlavePipeBase = collections.namedtuple("SlavePipeBase", ["identifier", "queue", "result"])

class SlavePipe(_SlavePipeBase):
    """
        Pipeline for master-slave communication
    """
    
    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret
    
class SyncMaster(object):
    """An abstract 'SyncMaster' object
        - During the **replication**, as the data parallel will trigger and callback of each module, all slave device should call `register(id)` and obtain a `SlavePipe` to comunicate with the master.
        
        - During the forward pass, master device invokes `run_master`, all messages from slave will be collected and passed to a registered callback.
            
        - After receiving the messages, the master device should gather the information and determine to message passed back to each slave devices.
    """
    def __init__(self, master_callback):
        """
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False
    
    def __getstate__(self) -> dict[str, Any]:
        return {
            "master_callback": self._master_callback
        }
        
    def __setstate__(self, state) -> None:
        self.__init__(state["master_callback"])

    def register_slave(self, identifier) -> SlavePipe:
        """
            Register a slave device.
            Args:
                Identifier: an identifier, usually the device id.
        """
        if self._activated:
            assert self._queue.empty(), "queue is not clean before next initialization"
            self._activated = False
            self._registry.clear()
        
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)
    
    def run_master(self, master_msg):
        """
            Main entry for the master device in each forward pass
            The message were first collected from each devices (include the master divice) and then an callback will be invoked to compute the message to be sent back to each devices (including the master device).
        """
        self._activated = True
        
        intermediates: List[Tuple[int, Any]] = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]
    
    @property
    def nr_slaves(self):
        return len(self._registry)

        
"""
    The Hidden PyTorch DataParallel Flow

    1. Model Replication Phase
    # When you do this:
    model = DataParallelWithCallback(sync_bn_model, device_ids=[0, 1, 2, 3])

    PyTorch internally:
    - Replicates your model to each GPU device
    - Calls __data_parallel_replicate__ on each module copy (lines 95-103 in batchnorm.py)
    - Device 0 (master): Gets the SyncMaster object
    - Other devices (slaves): Register with master and get SlavePipe objects

    2. Forward Pass Synchronization
    During each forward pass, automatically:

    # Line 80-83 in batchnorm.py shows the hidden branching:
    if self._parallel_id == 0:  # Master device
        mean, inv_std = self._sync_master.run_master(child_stats)
    else:  # Slave devices  
        mean, inv_std = self._slave_pipe.run_slave(child_stats)

    3. The Hidden Thread Orchestration

    What the user sees:
    output = model(input_batch)  # Looks like a simple forward pass

    What actually happens behind the scenes:
    1. DataParallel splits input to multiple devices
    2. Each device computes local batch statistics (sum, sum_of_squares, count)
    3. Slave threads call run_slave() → blocks on FutureResult.get()
    4. Master thread calls run_master() → collects all statistics → computes global mean/std →
    calls FutureResult.put() on each slave
    5. Slave threads wake up → receive global statistics → continue normalization
    6. DataParallel gathers outputs from all devices

    4. Thread Creation is Automatic

    PyTorch's DataParallel automatically creates threads for each GPU:
    - Each GPU gets its own thread
    - Your code runs concurrently on multiple threads
    - The synchronization happens transparently

    5. Visual Flow

    User Code: model(input)
        ↓
    DataParallel: Split input → [GPU0, GPU1, GPU2, GPU3]
        ↓
    Threads:    [Thread0, Thread1, Thread2, Thread3]
        ↓
    BatchNorm:  Compute local stats on each thread
        ↓
    Thread1-3:  call run_slave() → BLOCK on FutureResult.get()
    Thread0:    call run_master() → collect stats → compute global → FutureResult.put()
        ↓
    Thread1-3:  UNBLOCK → receive global stats → continue
        ↓
    DataParallel: Gather outputs from all threads
        ↓
    User Code: Receives final output

    Key Point: You never explicitly create threads or call FutureResult methods. PyTorch's
    DataParallel infrastructure handles all the threading, and the synchronized BatchNorm
    hooks into this system using the callback mechanism (__data_parallel_replicate__).

    This is why it's "hidden" - PyTorch abstracts away all the complex multi-threading
    synchronization, making it appear like a simple forward pass to the user.

    ──────────────────

"""