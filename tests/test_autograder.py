import json
import tempfile
import unittest
from pathlib import Path

from nxautograder import NXDrawingAutograder, PDFAnalysisResult


class TestNXDrawingAutograder(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = NXDrawingAutograder()

    def test_load_rubric_uses_requested_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rubric_path = Path(tmp) / "rubric.json"
            rubric_path.write_text(
                json.dumps(
                    {
                        "drawing-vector": {"pageSize": "A3", "material": "crs"},
                        "drawing-raster": {"pageSize": "B", "material": ""},
                    }
                ),
                encoding="utf-8",
            )
            rubric = self.grader.load_rubric(str(rubric_path), "drawing-vector")
        self.assertEqual(rubric["pageSize"], "A3")
        self.assertEqual(rubric["material"], "crs")

    def test_compare_to_rubric_returns_expected_shape(self) -> None:
        analysis = PDFAnalysisResult(
            pdf_type="vector",
            text="A3 MATERIAL: CRS X.X X.XX THIRD ANGLE ASME",
            page_size="A3",
            material="crs",
            dimensions_found=[1.5, 2.0, 4.0, 11.0],
            title_block={"value": True, "confidence": 5, "missing": []},
            notes={"value": True, "confidence": 5},
            tolerances={"value": True, "confidence": 5},
            projection={"value": True, "confidence": 5, "type": "third"},
            standard={"value": True, "confidence": 5, "type": "ASME"},
            labels=[],
        )
        rubric = {
            "pageSize": "A3",
            "material": "crs",
            "dimensions": [17, [[2.0, 4], [1.5, 0.75], 11]],
        }
        results = self.grader.compare_to_rubric(analysis, rubric)

        self.assertTrue(results["size"]["value"])
        self.assertTrue(results["material"]["value"])
        self.assertEqual(results["missingDims"]["count"], 2)
        self.assertEqual(results["missingDims"]["values"], [17.0, 0.75])

    def test_title_block_and_tolerance_variations(self) -> None:
        text = (
            "DATE DRAWN BY CHECKED BY APPROVED BY TITLE SHEET SIZE SHEET SCALE\n"
            "MATERIAL: CRS SURFACE FINISH 32 +/-"
        )
        tb = self.grader._check_title_block(text)
        tol = self.grader._check_tolerances(text)
        notes = self.grader._check_notes_block(text)
        self.assertTrue(tb["value"])
        self.assertTrue(tol["value"])
        self.assertTrue(notes["value"])


if __name__ == "__main__":
    unittest.main()
