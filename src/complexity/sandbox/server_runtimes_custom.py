# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import json
import logging
import multiprocessing
import os
import random
import resource
import psutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from os import path as osp
from typing import Any, List, Optional, Tuple

from filelock import FileLock

logger = logging.getLogger()

_fork_server: Optional["ForkServer"] = None
_fork_server_lock = threading.Lock()


class Executor(Enum):
    BUBBLEWRAP = 0
    FORK = 1
    BUBBLEWRAPUNLOCK = 2


@dataclass
class ResourceLimits:
    # Safe defaults
    memory: Optional[int] = int(2e9)
    tmpfs_size: Optional[int] = int(1e9)
    cpu_time: Optional[int] = None


class JSONConnection:
    """
    Simple Connection wrapper, but send() and recv() use JSON serialization to
    prevent side-effects when unpickling.
    """

    def __init__(self, c: Connection):
        self._c = c

    def send(self, obj: Any):
        self._c.send_bytes(json.dumps(obj).encode("utf8"))

    def recv(self) -> Any:
        # Putting some arbitrary limit here to prevent DoS.
        return json.loads(self._c.recv_bytes(20 * 1024 * 1024))

    def poll(self, timeout=0.0):
        return self._c.poll(timeout)

    def close(self):
        self._c.close()

    def fileno(self):
        return self._c.fileno()

    @property
    def closed(self):
        return self._c.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

def is_pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


class ForkServer:
    """
    Fork server for sandboxed program execution.

    Use spawn() to execute a program, specified by a command-line. It will
    return a handle and two file descriptors for sending input and obtaining
    output from the command, respectively. Commands need to be tailored to the
    input/output format expected by the call site.
    The command-line will be extended by two position arguments, corresponding
    to a numeric file descriptor to read input from and a numeric file
    descriptor to write output to.
    """

    # assign a cpu to the main process and avoid this cpu for the child processes ?
    # and therefore make sure the threads run on the main cpu only
    # further distinguish the cpus among the different parts of the sandbox
    # see if we can further split ram etc
    # detect the peaks ! make a measure for that :)
    # leverage the cprofilerwithin with the benchmarking part :)
    # couple all of that with a filtering at the end to remove peaks ? careful about that
    # do we correctly split the memory of course 


    def __init__(
        self, 
        cpu_id_list: Optional[List[int]] = None,
    ):
        """
        main_cpu_id is for the current process that is instanciating this forkserver class
        run_cpu_id is for the _run process, that is launched by this forkserver class
        """
        print('INIT FORKSERVER')

        # # we check that the forkserver is in the main cpu
        # if main_cpu_id is not None:
        #     # check that we are indeed on the right device when instanciating this class
        #     current_process = psutil.Process(os.getpid())
        #     assert len(current_process.cpu_affinity()) == len(main_cpu_id)
        #     assert current_process.cpu_affinity() == main_cpu_id

        # choose the start method of the child processes: spawn is I believe the most independent one (that is to say child processes inherit from the bare minimum)
        ctx = multiprocessing.get_context("spawn")

        # # the two ends of the connection pipe: with duplex false, left can only be used to receive, and right only to send
        # read_pipe, write_pipe = ctx.Pipe(duplex=False) 
        # the two ends of the connection pipe: with duplex false, left can only be used to receive, and right only to send

        # communicate with the runner
        pipe_run_1, pipe_run_2 = ctx.Pipe(duplex=True) 

        # we create a process that run the function _run
        # this is the process that is listening through the pipe to launch child processes with bubblewrap
        # we will call this process the launcher process, by opposition to the sandbox processes, being created by the launcher process
        self.process = ctx.Process(
            target=_run,
            args=(
                # os.getpid(),
                # vpid,
                pipe_run_2,
                # pipe_bubblewrap_2,
                # main_cpu_id,
                cpu_id_list,
                # timeout,
            ),
        )
        self.process.start()

        self.used_pid_list = [self.process.pid]
        assert is_pid_alive(self.process.pid)
        self.used_vpid_list = []

        # we close the pipe ?
        # read_pipe.close()
        
        self.lock = threading.Lock()
        self.next_vpid = 1 # process handles
        self.pipe_run_1 = pipe_run_1
        # self.pipe_buublewrap_1 = pipe_buublewrap_1

        self.pipe_list = [
            pipe_run_1, pipe_run_2,
            # pipe_buublewrap_1, pipe_bubblewrap_2
        ]
        self.rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))
        # self.main_cpu_id = main_cpu_id

    @staticmethod
    def global_instance() -> "ForkServer":
        raise Exception()

    def spawn(
        self,
        cmd: List[str],
        env: Optional[dict[str, str]] = None,
        executor: Executor = Executor.BUBBLEWRAP,
        rlimits: Optional[ResourceLimits] = None,
        timeout: int = 30,
        retries: int = 10,
        cpu_id_list: Optional[List[int]] = None,
    ) -> Tuple[int, Connection, JSONConnection]:
        """
        Launches a sandboxed process.

        If the process fails to start or does not send a canary message (see
        src/eval/sandbox/runners/python_tests.py for an example) within the
        specified timeout, retries will be attempted as specified.

        Returns a vPID that can be passed to ForkServer.kill() as well as
        multiprocessing.Connection objects to communicate with the sandboxed
        process (input writing and output reading ends).
        """
        if rlimits is None:
            rlimits = ResourceLimits()

        # for retry in range(max(retries + 1, 1)):
        #     if retry > 0:
        #         # exponential backoff + jitter
        #         t = 0.1 * (1.5 ** (retry + self.rng.random()))
        #         time.sleep(t)

        # create some pipes that will be used to communicate with the sandbox process
        input_r, input_w = multiprocessing.Pipe(duplex=False)
        output_r, output_w = multiprocessing.Pipe(duplex=False)

        # communicate with bubblewrap
        pipe_buublewrap_1, pipe_bubblewrap_2 = multiprocessing.Pipe(duplex=True) 
        self.pipe_buublewrap_1 = pipe_buublewrap_1

        # we save these connections to correctly close them later
        self.pipe_list.extend([
            input_r, input_w, output_r, output_w,
            pipe_buublewrap_1, pipe_bubblewrap_2
        ])

        if env is None:
            env = dict(os.environ)

        # here we go ask the launcher process to create a sandbox process
        with self.lock:
            vpid = self.next_vpid
            self.used_vpid_list.append(vpid)
            self.next_vpid += 1
            self.pipe_run_1.send(
                (vpid, "spawn_wo_fork", (executor, cmd, env, input_r, output_w, rlimits, cpu_id_list, pipe_bubblewrap_2))
            )

        while True:
            if not self.pipe_buublewrap_1.poll(timeout=30):
                print('#######\n#######\n#######\n#######\n#######\nISSUE STOP FORKSERVER')
                break

            data = self.pipe_buublewrap_1.recv()

            if 'pid' in data:
                self.used_pid_list.append(data[1])
                assert is_pid_alive(data[1])
                break

        output_rj = JSONConnection(output_r)

        return None, input_w, output_rj

    def kill(self, vpid: int):
        raise Exception()

    def stop(self):
        # that's the mother function that will kill everything that was instanciated from it
        # use it to correctly kill everything on timeout
        # in other words, that's the master killer
        with self.lock:
            self.pipe_run_1.send((0, "exit", None))

        while True:
            if not self.pipe_run_1.poll(timeout=10):
                print('#######\n#######\n#######\n#######\n#######\nISSUE STOP FORKSERVER')
                break

            data = self.pipe_run_1.recv()
            if data == 'you can kill':
                break

        self.process.kill()

        for pipe in self.pipe_list:
            pipe.close()

        # for vpid in self.used_vpid_list:
        #     os.killpg(vpid, signal.SIGKILL)
        print('killing processes', self.used_pid_list)
        for pid in self.used_pid_list:
            if is_pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

        time.sleep(3)

def _run(cmd_pipe, cpu_id_list, timeout=300000):
    # a process, not assigned to a specific cpu for now
    logger.info("Execution server: process started")

    # os.setpgid(os.getpid(), vpid)

    # set the cpu affinity
    
    if (cpu_id_list is not None):
        import psutil
        current_process = psutil.Process(os.getpid())
        current_process.cpu_affinity(cpu_id_list)
        
    rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))

    try:
        # pids = {}  # vpid -> pid
        bubblewrap_pid, process, status_r, status_w = None, None, None, None

        while True:
            if not cmd_pipe.poll(timeout=timeout):
                break
        
            vpid, command, extra = cmd_pipe.recv()

            if command == "spawn_wo_fork":
                executor, cmd, env, input_r, output_w, rlimits, sandbox_cpu_id_list, pipe_bubblewrap_2 = extra

                bubblewrap_pid, process, status_r, status_w = execute_bwrap_unlock(
                    vpid,
                    cmd,
                    env,
                    input_r,
                    output_w,
                    rlimits,
                    sandbox_cpu_id_list,
                    rng.randint(0, 10000),
                    pipe_bubblewrap_2,
                )

            elif command == "exit":
                if process is not None:
                    process.kill()
                    process.wait()
                if status_r is not None:
                    status_r.close()
                if status_w is not None:
                    status_w.close()
                break

            else:
                raise RuntimeError(
                    "Execution server: unknown command %s" % str(command)
                )
    except EOFError:
        logger.error("Execution server: main process died!")
        traceback.print_exc()

    finally:
        # for pid in pids.values():
        #     try:
        #         os.killpg(pid, signal.SIGKILL)
        #     except Exception:
        #         traceback.print_exc()

        try:
            process.kill()
            process.wait()
        except:
            pass

        try:
            status_r.close()
        except:
            pass

        try:
            status_w.close()
        except:
            pass

        cmd_pipe.send('you can kill')

def execute_bwrap_unlock(
    vpid,
    cmd: List[str],
    env: dict[str, str],
    input_r: Connection,
    output_w: Connection,
    rlimits: ResourceLimits,
    sandbox_cpu_id_list,
    seed: int,
    pipe_bubblewrap_2,
):
    def set_rlimit():
        if sandbox_cpu_id_list is not None:
            import psutil
            current_process = psutil.Process(os.getpid())
            current_process.cpu_affinity(sandbox_cpu_id_list)

        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

    # to be used to check whether the bubblewrap could start
    status_r, status_w = multiprocessing.Pipe(duplex=False)

    args = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
    ]
    if rlimits.tmpfs_size is not None:
        args += ["--size", str(rlimits.tmpfs_size)]

    args += [
        "--tmpfs",
        "/tmp",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--dir",
        "/tmp/sandbox",
        "--chdir",
        "/tmp/sandbox",
        "--info-fd",
        str(status_w.fileno()),
    ]

    args += [
        *cmd,
        str(input_r.fileno()),
        str(output_w.fileno()),
    ]

    # # Under heavy system load (kernel load, not CPU load), bubblewrap can take a
    # # long time to start. When launching many sandboxes in parallel, bubblewrap
    # # itself is a main cause of high load, unfortunately. Here, we attempt to
    # # linearlize sandboxes starting up in order to keep overall startup times
    # # low. We do this at the machine level with lock files and will warn the
    # # user if startup times are very high.
    # # We can isolate sandbox creation time with `--info-fd`: bubblewrap will
    # # write to it after launching the sandboxed process.
    # lockfile = osp.join(os.environ.get("TMPDIR", "/dev/shm"), "bwrap.lock")
    
    # Allow a handful concurrent sandbox launches

    # supposed to be 4
    # n_locks = 20
    random.seed(seed)

    # def get_lock():
    #     locks = [FileLock(lockfile + f".{i}", mode=0o777) for i in range(n_locks)]
    #     while True:
    #         lock = random.choice(locks)
    #         try:
    #             return lock.acquire(timeout=0.01)
    #         except BaseException:
    #             pass

    # start = time.perf_counter()
    # with get_lock():
    #     elapsed = time.perf_counter() - start
    #     if elapsed > 10:
    #         logger.warning(f"Waited {elapsed:.02f}s for bubblewrap lock")

    process = subprocess.Popen(
        args,
        close_fds=True,
        pass_fds=(input_r.fileno(), output_w.fileno(), status_w.fileno()),
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        env=env,
        preexec_fn=set_rlimit,
    )

    bubblewrap_pid = process.pid
    pipe_bubblewrap_2.send(('pid', bubblewrap_pid))
    # os.setpgid(bubblewrap_pid, vpid)

    # # Hold the lock for up to one second while bubblerwap is starting up,
    # # hopefully lessening the load on the kernel.
    # start = time.perf_counter()

    if not status_r.poll(1):
        # Keep waiting for bubblewrap to start and report if it's taking a very
        # long time.
        status_r.poll()

    #     elapsed = time.perf_counter() - start
    #     if elapsed > 10:
    #         logger.warning(f"bubblewrap took {elapsed:.02f}s to start")

    try:
        stdout, stderr = process.communicate()

    except BaseException:
        process.kill()
        process.wait()

        return bubblewrap_pid, process, status_r, status_w

    returncode = process.poll()

    try:
        output_w.send_bytes(
            json.dumps(
                {
                    "done": True,
                    "returncode": returncode,
                    "stderr": stderr.decode(errors="replace"),
                }
            ).encode("utf8")
        )
    except BrokenPipeError:
        # Most likely due to premature aborts (the caller stopped listening);
        # ignore this
        pass
    except Exception:
        logger.exception("Error sending 'done' message")

    return bubblewrap_pid, process, status_r, status_w