from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import resume_targeted_pt_v2_disk as r


class DiskRecoveryTests(unittest.TestCase):
    def test_all_cache_paths_are_private_children_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            env = r.cache_environment(out)
            self.assertEqual(set(env), set(r.CACHE_DIRS))
            self.assertEqual(r.cache_environment(out), env)
            for path in env.values():
                value = Path(path)
                self.assertEqual(value.parent, out / "runtime_cache_disk_v1")
                self.assertEqual(value.stat().st_mode & 0o777, 0o700)

    def test_refuse_symlink_root_without_touching_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            target = out / "other"
            target.mkdir()
            (out / "runtime_cache_disk_v1").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                r.cache_environment(out)
            self.assertEqual(list(target.iterdir()), [])

    def test_refuse_symlink_cache_without_touching_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / "runtime_cache_disk_v1"
            root.mkdir()
            target = out / "other"
            target.mkdir()
            (root / "triton").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                r.cache_environment(out)
            self.assertEqual(list(target.iterdir()), [])

    def test_storage_requires_different_device(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(Path, "home", return_value=out):
                with self.assertRaisesRegex(ValueError, "separate filesystem"):
                    r.verify_storage(out)

    def test_storage_requires_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            other_home = Mock()
            other_home.stat.return_value.st_dev = -1
            with patch.object(Path, "home", return_value=other_home):
                with patch.object(r.shutil, "disk_usage", return_value=Mock(free=1024)):
                    with self.assertRaisesRegex(ValueError, "20 GiB"):
                        r.verify_storage(out)
                with patch.object(r.shutil, "disk_usage", return_value=Mock(free=30 * 1024 ** 3)):
                    self.assertEqual(r.verify_storage(out)["free_bytes"], 30 * 1024 ** 3)


if __name__ == "__main__":
    unittest.main()
