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

A script that runs its code at module level after the app (Isaac Lab's style) calls
`guard(app)` right after the app starts instead: an exception that escapes the script then ends
the process with its status through `close`, from Python's excepthook, before the app's atexit
close runs. Python calls no excepthook for a `SystemExit`, so the script's `__main__` calls
`run(main)`, which maps any exception out of `main`, a `SystemExit` with a status among them.
"""
import logging
import os
import sys
import traceback

INTERRUPTED = 130  # 128 + SIGINT, a shell's status for a run stopped with Ctrl-C
_guarded = []  # the app guard() was given, for run()


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
    with `failed` as its status (True is 1), through `app.close(exit_code=...)` (none with no
    app) and, should that not end it, `_exit`."""
    if not failed:
        app.close()
        return
    code = 1 if failed is True else int(failed)
    logging.shutdown()
    _flush()
    try:
        if app is not None:
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



def guard(app=None, _exit=os._exit):
    """End the process with a failed status when an exception escapes the script: for a script
    whose code runs at module level after its app (Isaac Lab's style), or a wrapper that runs
    such a script. Python's excepthook prints the traceback as usual, and then this ends the
    process at once, before the app's atexit close could end it with 0: `close(app, code)` with
    1, or INTERRUPTED for Ctrl-C. With no app (a wrapper that never holds it) the end is a flush
    and `_exit(code)`, without Kit's cleanup. A script that ends normally still calls
    `app.close()`. A SystemExit that escapes is not seen (Python calls no excepthook for it):
    `run(main)` maps it, and any other exception out of `main`."""
    _guarded[:] = [app]
    hook = sys.excepthook

    def excepthook(kind, value, tb):
        hook(kind, value, tb)
        close(app, INTERRUPTED if issubclass(kind, KeyboardInterrupt) else 1, _exit)

    sys.excepthook = excepthook


def run(main, *args, **kwargs):
    """`main(...)`, for a script's `__main__`: an exception out of it, a `SystemExit` with a
    status among them (`raise SystemExit("why")` is 1), ends the process with its status
    (`code_of`, then `close` on the app `guard` was given, or none). Returns what `main`
    returns when it ends normally or with `SystemExit(0)`; closing the app on that path is the
    script's, as before."""
    try:
        return main(*args, **kwargs)
    except BaseException as e:
        code = code_of(e)
        if not code:
            return None
        close(_guarded[0] if _guarded else None, code)
