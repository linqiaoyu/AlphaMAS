import unittest

from cli.models import AnalystType
from cli.utils import ordered_selected_analyst_keys


class CliAnalystOrderTests(unittest.TestCase):
    def test_preserves_canonical_workflow_order(self):
        selected = [AnalystType.NEWS, AnalystType.MARKET, AnalystType.FUNDAMENTALS]
        self.assertEqual(
            ordered_selected_analyst_keys(selected),
            ["market", "news", "fundamentals"],
        )


if __name__ == "__main__":
    unittest.main()
