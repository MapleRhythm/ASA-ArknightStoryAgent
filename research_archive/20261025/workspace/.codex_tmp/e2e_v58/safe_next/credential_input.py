"""Credential input for a future run. Must be used with ssh -T, never ssh -tt."""
import os
import select
import stat


def read_secret_pipe(stream, timeout=30):
    """Refuse terminals/regular files before reading. Never echo or persist data."""
    fd = stream.fileno()
    if stream.isatty() or os.isatty(fd):
        raise ValueError("TTY credential input is disabled; use a non-PTY pipe")
    mode = os.fstat(fd).st_mode
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
        raise ValueError("Credential input must be a pipe or socket")
    value = bytearray()
    while True:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise TimeoutError("Credential input timed out")
        byte = os.read(fd, 1)
        if not byte or byte == b"\n":
            break
        value.extend(byte)
        if len(value) > 512:
            raise ValueError("Credential input too long")
    try:
        secret = value.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("Invalid credential input") from None
    finally:
        value[:] = b"\0" * len(value)
    if not secret or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in secret):
        raise ValueError("Invalid credential input")
    return secret
