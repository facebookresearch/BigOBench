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
import time

logger = logging.getLogger()

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


class LightForkServer:
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


    def __init__(self, main_cpu_id=None, launcher_cpu_id=None):
        print('INIT FORKSERVER')

        if main_cpu_id is not None:
            import psutil
            pid = os.getpid()
            current_process = psutil.Process(pid)
            assert len(current_process.cpu_affinity()) == len(main_cpu_id)
            assert current_process.cpu_affinity() == main_cpu_id

        # choose the start method of the child processes: spawn is I believe the most independent one (that is to say child processes inherit from the bare minimum)
        ctx = multiprocessing.get_context("spawn")

        # the two ends of the connection pipe: with duplex false, left can only be used to receive, and right only to send
        read_pipe, write_pipe = ctx.Pipe(duplex=False) 

        # we create a process that run the function _run
        # this is the process that is listening through the pipe to launch child processes with bubblewrap
        # we will call this process the launcher process, by opposition to the sandbox processes, being created by the launcher process
        self.process = ctx.Process(
            target=_run,
            args=(
                read_pipe,
                main_cpu_id,
                launcher_cpu_id,
            ),
        )
        self.process.start()        
        self.pipe = write_pipe
        self.pipe_list = [read_pipe, write_pipe]
        self.rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))
        self.main_cpu_id = main_cpu_id

    def spawn(
        self,
        cmd: List[str],
        env: Optional[dict[str, str]] = None,
        executor: Executor = Executor.BUBBLEWRAP,
        rlimits: Optional[ResourceLimits] = None,
        timeout: int = 30,
        retries: int = 10,
        cpu_id=None,
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

        input_r, input_w = multiprocessing.Pipe(duplex=False)
        output_r, output_w = multiprocessing.Pipe(duplex=False)

        self.pipe_list.extend([
            input_r, input_w, output_r, output_w
        ])

        if env is None:
            env = dict(os.environ)

        self.pipe.send(
            ("spawn_wo_fork", (executor, cmd, env, input_r, output_w, rlimits, cpu_id))
        )

        output_rj = JSONConnection(output_r)

        return input_w, output_rj

    def stop(self):
        self.pipe.send((0, "exit", None))
        time.sleep(3)

        for pipe in self.pipe_list:
            pipe.close()
            
        self.process.shutdown()
        time.sleep(3)


def _run(cmd_pipe, main_cpu_id = None, launcher_cpu_id = None):
    # a process, not assigned to a specific cpu for now
    logger.info("Execution server: process started")

    import psutil
    if (launcher_cpu_id is not None) and (main_cpu_id is not None):
        import psutil
        pid = os.getpid()
        current_process = psutil.Process(pid)
        assert len(current_process.cpu_affinity()) == len(main_cpu_id)
        assert current_process.cpu_affinity() == main_cpu_id

        current_process.cpu_affinity(launcher_cpu_id)
        assert current_process.cpu_affinity() == launcher_cpu_id
        
    rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))

    try:
        while True:
            command, extra = cmd_pipe.recv()

            if command == "spawn_wo_fork":
                executor, cmd, env, input_r, output_w, rlimits, cpu_id = extra

                # we fork, creating a copy of the parent, child_pid is 0 is the child process, and the child id in the parent process
                # child_pid = os.fork()

                # if child_pid == 0:
                #     pid = os.getpid()
                #     os.setpgid(pid, pid)

                if (cpu_id is not None):
                    current_process = psutil.Process(pid)
                    assert len(current_process.cpu_affinity()) == len(launcher_cpu_id)
                    assert current_process.cpu_affinity() == launcher_cpu_id
                    current_process.cpu_affinity(cpu_id)
                    assert current_process.cpu_affinity() == cpu_id

                # pids = {}
                match executor:
                    case Executor.BUBBLEWRAP:
                        execute_bwrap(
                            cmd,
                            env,
                            input_r,
                            output_w,
                            rlimits,
                            seed=rng.randint(0, 10000),
                        )
                    case Executor.BUBBLEWRAPUNLOCK:
                        execute_bwrap_unlock(
                            cmd,
                            env,
                            input_r,
                            output_w,
                            rlimits,
                            seed=rng.randint(0, 10000),
                        )
                    case Executor.FORK:
                        execute_fork(
                            cmd,
                            env,
                            input_r,
                            output_w,
                            rlimits,
                        )
                    case _:
                        logger.error(f"Unknown executor: {executor}")
                        output_w.send_bytes(
                            json.dumps({"error": "unknown executor"}).encode("utf8")
                        )

            elif command == "exit":
                break

            else:
                raise RuntimeError(
                    "Execution server: unknown command %s" % str(command)
                )

    except EOFError:
        logger.error("Execution server: main process died!")
        traceback.print_exc()

def execute_bwrap(
    cmd: List[str],
    env: dict[str, str],
    input_r: Connection,
    output_w: Connection,
    rlimits: ResourceLimits,
    seed: int,
):
    def set_rlimit():
        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

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

    # Under heavy system load (kernel load, not CPU load), bubblewrap can take a
    # long time to start. When launching many sandboxes in parallel, bubblewrap
    # itself is a main cause of high load, unfortunately. Here, we attempt to
    # linearlize sandboxes starting up in order to keep overall startup times
    # low. We do this at the machine level with lock files and will warn the
    # user if startup times are very high.
    # We can isolate sandbox creation time with `--info-fd`: bubblewrap will
    # write to it after launching the sandboxed process.
    lockfile = osp.join(os.environ.get("TMPDIR", "/dev/shm"), "bwrap.lock")
    
    # Allow a handful concurrent sandbox launches

    # supposed to be 4
    n_locks = 20
    random.seed(seed)

    def get_lock():
        locks = [FileLock(lockfile + f".{i}", mode=0o777) for i in range(n_locks)]
        while True:
            lock = random.choice(locks)
            try:
                return lock.acquire(timeout=0.01)
            except BaseException:
                pass

    start = time.perf_counter()
    with get_lock():
        elapsed = time.perf_counter() - start
        if elapsed > 10:
            logger.warning(f"Waited {elapsed:.02f}s for bubblewrap lock")
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

        # Hold the lock for up to one second while bubblerwap is starting up,
        # hopefully lessening the load on the kernel.
        start = time.perf_counter()
        r = status_r.poll(1)

    if not r:
        # Keep waiting for bubblewrap to start and report if it's taking a very
        # long time.
        r = status_r.poll()
        elapsed = time.perf_counter() - start
        if elapsed > 10:
            logger.warning(f"bubblewrap took {elapsed:.02f}s to start")

    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise

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

def execute_bwrap_unlock(
    cmd: List[str],
    env: dict[str, str],
    input_r: Connection,
    output_w: Connection,
    rlimits: ResourceLimits,
    seed: int,
):
    def set_rlimit():
        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

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

    # Under heavy system load (kernel load, not CPU load), bubblewrap can take a
    # long time to start. When launching many sandboxes in parallel, bubblewrap
    # itself is a main cause of high load, unfortunately. Here, we attempt to
    # linearlize sandboxes starting up in order to keep overall startup times
    # low. We do this at the machine level with lock files and will warn the
    # user if startup times are very high.
    # We can isolate sandbox creation time with `--info-fd`: bubblewrap will
    # write to it after launching the sandboxed process.
    # lockfile = osp.join(os.environ.get("TMPDIR", "/dev/shm"), "bwrap.lock")
    
    # Allow a handful concurrent sandbox launches
    # n_locks = 4
    random.seed(seed)

    # def get_lock():
    #     locks = [FileLock(lockfile + f".{i}", mode=0o777) for i in range(n_locks)]
    #     while True:
    #         lock = random.choice(locks)
    #         try:
    #             return lock.acquire(timeout=0.01)
    #         except BaseException:
    #             pass

    start = time.perf_counter()
    # with get_lock():
    elapsed = time.perf_counter() - start
    if elapsed > 10:
        logger.warning(f"Waited {elapsed:.02f}s for bubblewrap lock")
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

    # Hold the lock for up to one second while bubblerwap is starting up,
    # hopefully lessening the load on the kernel.
    start = time.perf_counter()
    r = status_r.poll(1)

    if not r:
        # Keep waiting for bubblewrap to start and report if it's taking a very
        # long time.
        r = status_r.poll()
        elapsed = time.perf_counter() - start
        if elapsed > 10:
            logger.warning(f"bubblewrap took {elapsed:.02f}s to start")

    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise

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


def execute_fork(
    cmd: List[str],
    env: dict[str, str],
    input_r: Connection,
    output_w: Connection,
    rlimits: ResourceLimits,
):
    def set_rlimit():
        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

    status_r, status_w = multiprocessing.Pipe(duplex=False)

    if rlimits.tmpfs_size is not None:
        logger.warning("tempfs size cannot be enforced")

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = osp.join(tmpdir, "run")
        os.mkdir(cwd)
        tmpdir = osp.join(tmpdir, "tmp")
        os.mkdir(tmpdir)
        env["TMPDIR"] = tmpdir
        env["TEMP"] = tmpdir
        env["TMP"] = tmpdir

        process = subprocess.Popen(
            [*cmd, str(input_r.fileno()), str(output_w.fileno())],
            close_fds=True,
            pass_fds=(input_r.fileno(), output_w.fileno(), status_w.fileno()),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
            preexec_fn=set_rlimit,
            cwd=cwd,
        )

        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.kill()
            process.wait()
            raise

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
