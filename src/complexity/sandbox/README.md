# Sandboxing support

The present package offers sandboxing capabilities to execute untrusted
code. Sandboxing is currently implemented by [bubblewrap][1] and ensures
that the code executed does not have network access, does not have write
access to any directory but an in-memory scratch space, is not able to
observe concurrent processes, and does not exceed a memory usage limit.

Specific limits can be set using the [`ResourceLimits`][2] class.

## APIs

The API for sandboxing is split in two layers. A high-level interface
allows to run easily an arbitrary piece of Python code using the
[`Runner`][3] class or the [`sandbox.run`][3] function. Then, a low-level
but more flexible [`ForkServer`][4] class allows to spawn an arbitrary
executable and communicate with it using multiprocessing connection
objects.

Using the high-level API, sandboxing is as easy as

```python
import sandbox

# the lambda runs in a sanboxed environment
assert sandbox.run(lambda x: x + 1, 41) == 42
```


[1]: https://github.com/containers/bubblewrap
