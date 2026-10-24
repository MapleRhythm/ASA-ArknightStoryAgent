import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import deepseek_compare as d


class NetworkTests(unittest.TestCase):
    def test_http_auth_error_counts_as_reachable(self):
        fake = Mock()
        fake.open.side_effect = __import__("urllib.error").error.HTTPError(
            d.ENDPOINT.rsplit("/", 1)[0], 401, "auth", {}, None
        )
        with patch("urllib.request.build_opener", return_value=fake):
            self.assertEqual(d.network_preflight()["http_status"], 401)

    def test_connection_failure_is_bounded(self):
        fake = Mock()
        fake.open.side_effect = TimeoutError("connect")
        with patch("urllib.request.build_opener", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                d.network_preflight()


if __name__ == "__main__":
    unittest.main()
