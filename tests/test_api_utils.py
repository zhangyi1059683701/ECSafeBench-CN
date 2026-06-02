from __future__ import annotations

import unittest

from energy_chem_benchmark.api_utils import authorization_value


class AuthorizationValueTest(unittest.TestCase):
    def test_bearer_style(self) -> None:
        self.assertEqual(authorization_value("sk-test"), "Bearer sk-test")

    def test_raw_style_for_dmxapi(self) -> None:
        self.assertEqual(authorization_value("sk-test", "raw"), "sk-test")

    def test_invalid_style(self) -> None:
        with self.assertRaises(ValueError):
            authorization_value("sk-test", "unknown")


if __name__ == "__main__":
    unittest.main()
