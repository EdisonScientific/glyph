from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import tools.mg2_scorer.rescore_with_public_mg2 as rescore
from tools.mg2_scorer.rescore_with_public_mg2 import (
    configure_public_mg2_imports,
    convert_prediction_official,
    load_official_conversion_tools,
    prepare_for_public_mg2,
    score_samples_with_public_mg2,
)


class PublicMG2ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            configure_public_mg2_imports()
            from markushgrapher.utils.ocsr.utils_evaluation import get_scores
        except Exception as exc:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"public MG2 scorer is not importable: {exc}") from exc
        cls.get_scores = staticmethod(get_scores)

    def test_local_r_tags_are_adapted_to_rdkit_cxsmiles(self) -> None:
        converted = prepare_for_public_mg2("<r>R1</r>C")

        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertNotIn("<r>", converted)
        self.assertIn("$R1", converted)

        result = score_samples_with_public_mg2(
            [
                {
                    "sample_id": "local-exact",
                    "ground_truth": "<r>R1</r>C",
                    "prediction_cxsmiles": "<r>R1</r>C",
                }
            ],
            contract="generic",
            get_scores_fn=self.get_scores,
        )

        self.assertEqual(result["exact_counts"]["cxsmi_equality_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_total"], 1)
        self.assertTrue(result["samples"][0]["score"]["cxsmi_equality"])

    def test_exact_recall_miss_and_hallucinated_extra_verdicts(self) -> None:
        result = score_samples_with_public_mg2(
            [
                {
                    "sample_id": "exact",
                    "ground_truth": "*C |$R1;$|",
                    "prediction_cxsmiles": "*C |$R1;$|",
                },
                {
                    "sample_id": "recall-miss",
                    "ground_truth": "*C |$R1;$|",
                    "prediction_cxsmiles": "CC",
                },
                {
                    "sample_id": "hallucinated-extra",
                    "ground_truth": "CC",
                    "prediction_cxsmiles": "*CC |$R1;;$|",
                },
            ],
            contract="generic",
            get_scores_fn=self.get_scores,
        )

        verdicts = {row["sample_id"]: row["score"]["cxsmi_equality"] for row in result["samples"]}
        self.assertEqual(
            verdicts, {"exact": True, "recall-miss": False, "hallucinated-extra": False}
        )
        self.assertEqual(result["exact_counts"]["cxsmi_equality_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_total"], 3)
        self.assertAlmostEqual(result["exact_counts"]["cxsmi_equality_accuracy"], 1 / 3)

    def test_invalid_ground_truth_is_not_prefiltered(self) -> None:
        result = score_samples_with_public_mg2(
            [{"sample_id": "bad-gt", "ground_truth": "not a smiles", "prediction_cxsmiles": "CC"}],
            contract="generic",
            get_scores_fn=self.get_scores,
        )

        self.assertEqual(result["exact_counts"]["size"], 1)
        self.assertEqual(result["exact_counts"]["invalid_gt_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_total"], 0)
        self.assertIsNone(result["samples"][0]["score"])


class OfficialContractTests(unittest.TestCase):
    """Cover the official-contract prediction conversion + DS4SD-GT scoring."""

    # Row EP90115748NWA2_23_0 from the IP5-M reference greedy predictions: a
    # correct row under the official contract (cxsmi_equality True).
    WRAPPED_PREDICTION = (
        "<markush><cxsmi><r>SCH3</r>C(N)=NC1=N<r>Z</r>=<r>Y</r>C(<r>R4</r>)=<r>X</r>1"
        "</cxsmi><stable></stable></markush>"
    )
    DS4SD_GT = "[4*]C1=[5*]C(N=C(N)SC)=N[2*]=[3*]1 |$R4;;X;;;;;;;;Z;Y$|"

    @classmethod
    def setUpClass(cls) -> None:
        try:
            configure_public_mg2_imports()
            from markushgrapher.utils.ocsr.utils_evaluation import get_scores

            cls.tools = load_official_conversion_tools()
        except Exception as exc:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"public MG2 scorer is not importable: {exc}") from exc
        cls.get_scores = staticmethod(get_scores)

    def test_convert_opt_to_out_decodes_wrapped_prediction(self) -> None:
        tokenizer, abbreviation, fix_cxsmiles, chem = self.tools
        record = convert_prediction_official(
            self.WRAPPED_PREDICTION,
            tokenizer=tokenizer,
            abbreviation=abbreviation,
            fix_cxsmiles=fix_cxsmiles,
            chem=chem,
        )

        self.assertIsNone(record["failure"])
        self.assertEqual(record["final_prediction"], self.DS4SD_GT)

    def test_missing_closing_cxsmi_tag_is_a_conversion_failure(self) -> None:
        tokenizer, abbreviation, fix_cxsmiles, chem = self.tools
        record = convert_prediction_official(
            "<markush><cxsmi>CC (never closed",
            tokenizer=tokenizer,
            abbreviation=abbreviation,
            fix_cxsmiles=fix_cxsmiles,
            chem=chem,
        )

        self.assertEqual(record["failure"], "missing_cxsmi_payload")
        self.assertIsNone(record["final_prediction"])

    def test_official_contract_scores_correct_row_against_ds4sd_gt(self) -> None:
        result = score_samples_with_public_mg2(
            [{"sample_id": "EP90115748NWA2_23_0", "prediction": self.WRAPPED_PREDICTION}],
            contract="official",
            gt_by_id={"EP90115748NWA2_23_0": self.DS4SD_GT},
            conversion_tools=self.tools,
            get_scores_fn=self.get_scores,
        )

        self.assertEqual(result["exact_counts"]["cxsmi_equality_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_full_accuracy"], 1.0)
        self.assertEqual(result["exact_counts"]["conversion_failure_count"], 0)
        self.assertTrue(result["samples"][0]["score"]["cxsmi_equality"])

    def test_official_contract_retains_none_gt_as_incorrect(self) -> None:
        result = score_samples_with_public_mg2(
            [
                {"sample_id": "EP90115748NWA2_23_0", "prediction": self.WRAPPED_PREDICTION},
                {"sample_id": "none-gt-row", "prediction": self.WRAPPED_PREDICTION},
            ],
            contract="official",
            gt_by_id={"EP90115748NWA2_23_0": self.DS4SD_GT, "none-gt-row": None},
            conversion_tools=self.tools,
            get_scores_fn=self.get_scores,
        )

        # The None-GT row stays in the denominator but scores as incorrect.
        self.assertEqual(result["exact_counts"]["size"], 2)
        self.assertEqual(result["exact_counts"]["invalid_gt_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_count"], 1)
        self.assertEqual(result["exact_counts"]["cxsmi_equality_full_accuracy"], 0.5)
        self.assertIsNone(result["samples"][1]["score"])


class VendorPathThreadingTests(unittest.TestCase):
    """Regression: custom ``--vendor-*`` overrides thread through all loaders.

    Runs without the vendored MG2 package: only ``sys.path`` bookkeeping and
    the module-level configured-path state are exercised.
    """

    def setUp(self) -> None:
        self._saved_sys_path = list(sys.path)
        self._saved_default_grapher = rescore.DEFAULT_VENDOR_GRAPHER
        self._saved_default_generator = rescore.DEFAULT_VENDOR_GENERATOR
        self._saved_configured_grapher = rescore._CONFIGURED_GRAPHER
        self._saved_configured_generator = rescore._CONFIGURED_GENERATOR
        # Simulate a clean checkout where the *default* vendor dirs are absent.
        rescore.DEFAULT_VENDOR_GRAPHER = Path("/nonexistent/default/MarkushGrapher")
        rescore.DEFAULT_VENDOR_GENERATOR = Path("/nonexistent/default/MarkushGenerator")
        rescore._CONFIGURED_GRAPHER = None
        rescore._CONFIGURED_GENERATOR = None
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.custom_grapher = root / "custom" / "MarkushGrapher"
        self.custom_generator = root / "custom" / "MarkushGenerator"
        self.custom_grapher.mkdir(parents=True)
        self.custom_generator.mkdir(parents=True)

    def tearDown(self) -> None:
        sys.path[:] = self._saved_sys_path
        rescore.DEFAULT_VENDOR_GRAPHER = self._saved_default_grapher
        rescore.DEFAULT_VENDOR_GENERATOR = self._saved_default_generator
        rescore._CONFIGURED_GRAPHER = self._saved_configured_grapher
        rescore._CONFIGURED_GENERATOR = self._saved_configured_generator
        self._tmp.cleanup()

    def test_library_default_contract_is_official(self) -> None:
        # The module docstring and CLI both say the default contract is
        # 'official'; the library entry point must agree.
        default = inspect.signature(score_samples_with_public_mg2).parameters["contract"].default
        self.assertEqual(default, "official")

    def test_all_loaders_accept_vendor_overrides(self) -> None:
        for fn in (
            configure_public_mg2_imports,
            rescore._load_public_get_scores,
            rescore._load_rdkit_chem,
            load_official_conversion_tools,
            score_samples_with_public_mg2,
        ):
            params = inspect.signature(fn).parameters
            self.assertIn("grapher_path", params, fn.__name__)
            self.assertIn("generator_path", params, fn.__name__)

    def test_custom_vendor_dirs_only_do_not_reapply_defaults(self) -> None:
        # Configuring with explicit custom dirs must succeed even though the
        # defaults are absent...
        configure_public_mg2_imports(
            grapher_path=self.custom_grapher, generator_path=self.custom_generator
        )
        self.assertIn(str(self.custom_grapher), sys.path)
        self.assertIn(str(self.custom_generator), sys.path)

        # ...and a subsequent *no-argument* call (as a deep helper makes) must
        # reuse the configured custom dirs, not re-validate the absent defaults.
        try:
            configure_public_mg2_imports()
        except FileNotFoundError as exc:  # pragma: no cover - regression guard
            self.fail(f"no-argument reconfigure re-applied absent defaults: {exc}")
        self.assertNotIn(str(rescore.DEFAULT_VENDOR_GRAPHER), sys.path)

    def test_score_samples_threads_custom_vendor_dirs(self) -> None:
        captured: dict[str, object] = {}

        def fake_get_scores(gt, pred, train, **kwargs):
            captured["gt"] = list(gt)
            return ({"n": len(list(gt))}, {i: {"cxsmi_equality": True} for i in range(len(gt))})

        # Only custom dirs exist; a FileNotFoundError here would mean a loader
        # silently re-applied the absent default vendor paths.
        result = score_samples_with_public_mg2(
            [{"sample_id": "row0", "ground_truth": "CC", "prediction_cxsmiles": "CC"}],
            contract="generic",
            convert_local_r_tags=False,
            get_scores_fn=fake_get_scores,
            grapher_path=self.custom_grapher,
            generator_path=self.custom_generator,
        )
        self.assertEqual(result["exact_counts"]["cxsmi_equality_count"], 1)
        self.assertIn(str(self.custom_grapher), sys.path)


if __name__ == "__main__":
    unittest.main()
