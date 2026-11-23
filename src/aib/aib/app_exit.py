"""Ending a Kit app with the run's exit status (scensim issue 118).

`SimulationApp.close()` ends the process itself, with status 0: an entry point whose
`finally` calls it exits 0 whatever the `try` raised, and the exception's traceback is lost
with the interpreter. So an entry point records how its run ended and hands that here:

    app = SimulationApp(...)
    code = 0
    try:
        ...
    except BaseException as e:
        code = app_exit.code_of(e)    # the traceback on stderr; the status it maps to
    finally:
        ...                           # what the run writes either way
        app_exit.close(app, code)

The `try` starts right after the app: an exception that escapes the entry point ends it with
0 too, through the app's atexit close.

`close` calls `app.close()` for a run that passed. For a failed one it calls
`app.close(exit_code=code)`: Isaac 6.2's close runs its cleanup (replicator's queued writes,
the stage) and then ends the process with that status. Should the app's close return, refuse
the argument or raise, the run still ends with `os._exit(code)` after a flush. Pure: no Isaac
import.
"""
import logging
import os
import sys
import traceback

INTERRUPTED = 130  # 128 + SIGINT, a shell's status for a run stopped with Ctrl-C


def code_of(exc):
    """The exit status a run that raised `exc` ends with, its traceback written to stderr.

    SystemExit keeps its own status (None is 0, a message is 1, as Python's own exit); a
    KeyboardInterrupt is INTERRUPTED; anything else is 1."""
    if isinstance(exc, SystemExit):
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or 0
        print(exc.code, file=sys.stderr, flush=True)
        return 1
    if isinstance(exc, KeyboardInterrupt):
        print("interrupted", file=sys.stderr, flush=True)
        return INTERRUPTED
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    return 1


def close(app, failed, _exit=os._exit):
    """End the run: `app.close()` when it passed (`failed` false); otherwise end the process
    with `failed` as its status (True is 1), through `app.close(exit_code=...)` and, should that
    not end it, `_exit`."""
    if not failed:
        app.close()
        return
    code = 1 if failed is True else int(failed)
    logging.shutdown()
    _flush()
    try:
        app.close(exit_code=code)
    except TypeError:  # an app whose close takes no exit_code (before Isaac 6.2)
        pass
    except BaseException:
        traceback.print_exc()
    _flush()
    _exit(code)


def _flush():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):  # a closed or broken pipe: the status still goes out
            pass
