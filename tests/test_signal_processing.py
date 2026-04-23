import unittest

from tradingagents.graph.signal_processing import SignalProcessor


class SignalProcessingTests(unittest.TestCase):
    def setUp(self):
        self.processor = SignalProcessor(quick_thinking_llm=None)

    def test_extracts_explicit_rating_line(self):
        signal = "Rating: BUY\nExecutive Summary: Enter on pullbacks."
        self.assertEqual(self.processor.process_signal(signal), "BUY")

    def test_falls_back_to_final_transaction_proposal(self):
        signal = "Some discussion.\nFINAL TRANSACTION PROPOSAL: **SELL**"
        self.assertEqual(self.processor.process_signal(signal), "SELL")

    def test_defaults_to_hold_when_no_action_found(self):
        signal = "This report is inconclusive and contains no actionable label."
        self.assertEqual(self.processor.process_signal(signal), "HOLD")

    def test_ignores_unlabeled_action_words_in_body_text(self):
        signal = (
            "The discussion mentions BUY and SELL as examples, but there is no explicit rating."
        )
        self.assertEqual(self.processor.process_signal(signal), "HOLD")


if __name__ == "__main__":
    unittest.main()
