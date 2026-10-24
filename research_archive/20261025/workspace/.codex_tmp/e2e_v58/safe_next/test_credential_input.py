import io
import os
import pty
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from credential_input import read_secret_pipe


class Tests(unittest.TestCase):
    def pipe_value(self, value):
        rd, wr = os.pipe()
        os.write(wr, value)
        os.close(wr)
        with os.fdopen(rd, "rb", buffering=0) as stream:
            return read_secret_pipe(stream, timeout=0.1)

    def test_pipe_no_output(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(self.pipe_value(b"dummy-unit-test-only\n"), "dummy-unit-test-only")
        self.assertEqual(out.getvalue() + err.getvalue(), "")

    def test_actual_pty_refused_before_read(self):
        master, slave = pty.openpty()
        try:
            with os.fdopen(slave, "rb", buffering=0, closefd=False) as stream:
                with self.assertRaisesRegex(ValueError, "TTY"):
                    read_secret_pipe(stream, timeout=0.1)
        finally:
            os.close(slave)
            os.close(master)

    def test_regular_file_refused(self):
        with tempfile.TemporaryFile() as f:
            f.write(b"dummy\n")
            f.seek(0)
            with self.assertRaises(ValueError):
                read_secret_pipe(f)

    def test_empty_refused(self):
        with self.assertRaises(ValueError):
            self.pipe_value(b"\n")

    def test_whitespace_refused(self):
        with self.assertRaises(ValueError):
            self.pipe_value(b"dummy value\n")

    def test_long_refused(self):
        with self.assertRaises(ValueError):
            self.pipe_value(b"x" * 513 + b"\n")

    def test_timeout(self):
        rd, wr = os.pipe()
        try:
            with os.fdopen(rd, "rb", buffering=0, closefd=False) as stream:
                with self.assertRaises(TimeoutError):
                    read_secret_pipe(stream, timeout=0.001)
        finally:
            os.close(rd)
            os.close(wr)


if __name__ == "__main__":
    unittest.main()
