from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PDFAnalysisResult:
    pdf_type: str
    text: str
    page_size: str | None
    material: str | None
    dimensions_found: list[float]
    title_block: dict[str, Any]
    notes: dict[str, Any]
    tolerances: dict[str, Any]
    projection: dict[str, Any]
    standard: dict[str, Any]
    labels: list[dict[str, Any]] = field(default_factory=list)
    contour_count: int = 0
    vector_line_count: int = 0


class NXDrawingAutograder:
    TITLE_BLOCK_FIELDS = (
        "date",
        "drawn by",
        "checked by",
        "approved by",
        "title",
        "sheet size",
        "sheet scale",
    )

    def open_pdf(self, pdf_path: str) -> bytes:
        path = Path(pdf_path)
        if not path.exists() or path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return path.read_bytes()

    def is_vector_pdf(self, pdf_path: str) -> bool:
        pdf_bytes = self.open_pdf(pdf_path)
        text = pdf_bytes.decode("latin-1", errors="ignore")
        vector_ops = len(re.findall(r"[\s\n](?:m|l|c|v|y|h|re|S|s|f|B|b|Do)[\s\n]", text))
        images = len(re.findall(r"/Subtype\s*/Image", text))
        return vector_ops >= max(1, images)

    def load_rubric(self, json_path: str, rubric_key: str) -> dict[str, Any]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Rubric file not found: {json_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if rubric_key not in data:
            raise KeyError(f"Rubric key '{rubric_key}' not found in rubric JSON.")
        return data[rubric_key]

    def process_pdf(self, pdf_path: str) -> PDFAnalysisResult:
        if self.is_vector_pdf(pdf_path):
            return self._process_vector(pdf_path)
        return self._process_raster(pdf_path)

    def _process_vector(self, pdf_path: str) -> PDFAnalysisResult:
        text = self._extract_text_vector(pdf_path)
        vector_line_count = self._estimate_vector_line_count(pdf_path)
        return self._build_analysis("vector", text, vector_line_count=vector_line_count)

    def _process_raster(self, pdf_path: str) -> PDFAnalysisResult:
        text = self._extract_text_raster(pdf_path)
        contour_count = self._estimate_contours(pdf_path)
        return self._build_analysis("raster", text, contour_count=contour_count)

    def _build_analysis(
        self,
        pdf_type: str,
        text: str,
        contour_count: int = 0,
        vector_line_count: int = 0,
    ) -> PDFAnalysisResult:
        page_size = self._extract_page_size(text)
        material = self._extract_material(text)
        dimensions = self._extract_dimensions(text)
        title_block = self._check_title_block(text)
        notes = self._check_notes_block(text)
        tolerances = self._check_tolerances(text)
        projection = self._check_projection_symbol(text)
        standard = self._check_standard(text)
        labels = self.default_label_boxes()
        return PDFAnalysisResult(
            pdf_type=pdf_type,
            text=text,
            page_size=page_size,
            material=material,
            dimensions_found=dimensions,
            title_block=title_block,
            notes=notes,
            tolerances=tolerances,
            projection=projection,
            standard=standard,
            labels=labels,
            contour_count=contour_count,
            vector_line_count=vector_line_count,
        )

    def _extract_text_vector(self, pdf_path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(pdf_path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            raw = self.open_pdf(pdf_path).decode("latin-1", errors="ignore")
            candidates = re.findall(r"\(([^\)]{2,120})\)", raw)
            return "\n".join(candidates)

    def _extract_text_raster(self, pdf_path: str) -> str:
        try:
            from pdf2image import convert_from_path  # type: ignore
            import pytesseract  # type: ignore

            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not images:
                return ""
            return pytesseract.image_to_string(images[0])
        except Exception:
            return self._extract_text_vector(pdf_path)

    def _estimate_contours(self, pdf_path: str) -> int:
        try:
            from pdf2image import convert_from_path  # type: ignore
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            pages = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not pages:
                return 0
            image = np.array(pages[0].convert("L"))
            _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len(contours)
        except Exception:
            return 0

    def _estimate_vector_line_count(self, pdf_path: str) -> int:
        content = self.open_pdf(pdf_path).decode("latin-1", errors="ignore")
        return len(re.findall(r"[\s\n](?:m|l|re|S|s)[\s\n]", content))

    def _check_title_block(self, text: str) -> dict[str, Any]:
        normalized = text.lower()
        missing = [field for field in self.TITLE_BLOCK_FIELDS if field not in normalized]
        found = len(missing) == 0
        confidence = max(1, 5 - len(missing))
        return {"value": found, "confidence": confidence, "missing": missing}

    def _check_notes_block(self, text: str) -> dict[str, Any]:
        normalized = text.lower()
        material_ok = "material" in normalized
        finish_ok = ("surface finish" in normalized) or ("finish" in normalized)
        value = material_ok and finish_ok
        confidence = 5 if value else 3 if (material_ok or finish_ok) else 1
        return {"value": value, "confidence": confidence}

    def _check_tolerances(self, text: str) -> dict[str, Any]:
        normalized = text.lower()
        tolerance_tokens = [
            r"x\.x",
            r"x\.xx",
            r"\+/-",
            r"±",
            r"\+\\-",
            r"plus[-\s]?minus",
        ]
        hits = sum(1 for token in tolerance_tokens if re.search(token, normalized))
        value = hits > 0
        confidence = min(5, max(1, hits + 1))
        return {"value": value, "confidence": confidence}

    def _check_projection_symbol(self, text: str) -> dict[str, Any]:
        normalized = text.lower()
        if "third angle" in normalized or "3rd angle" in normalized:
            return {"value": True, "confidence": 5, "type": "third"}
        if "first angle" in normalized or "1st angle" in normalized:
            return {"value": True, "confidence": 5, "type": "first"}
        return {"value": False, "confidence": 1, "type": None}

    def _check_standard(self, text: str) -> dict[str, Any]:
        normalized = text.lower()
        if "asme" in normalized:
            return {"value": True, "confidence": 5, "type": "ASME"}
        if "iso" in normalized:
            return {"value": True, "confidence": 5, "type": "ISO"}
        return {"value": False, "confidence": 1, "type": None}

    def _extract_page_size(self, text: str) -> str | None:
        m = re.search(r"\b(a3|b)\b", text, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _extract_material(self, text: str) -> str | None:
        normalized = text.lower()
        if "crs" in normalized:
            return "crs"
        m = re.search(r"material\s*:?\s*([a-z0-9\-\s]+)", normalized)
        if not m:
            return None
        return m.group(1).strip().split()[0]

    def _extract_dimensions(self, text: str) -> list[float]:
        values = [float(v) for v in re.findall(r"(?<!\w)(?:\d+\.\d+|\d+)(?!\w)", text)]
        unique = []
        seen: set[float] = set()
        for value in values:
            if value not in seen:
                unique.append(value)
                seen.add(value)
        return unique

    def default_label_boxes(self) -> list[dict[str, Any]]:
        return [
            {"label": "title_block", "x": 0.72, "y": 0.78, "w": 0.26, "h": 0.2},
            {"label": "notes_block", "x": 0.72, "y": 0.52, "w": 0.26, "h": 0.24},
            {"label": "projection_symbol", "x": 0.64, "y": 0.78, "w": 0.07, "h": 0.1},
        ]

    def display_with_boxes(
        self,
        pdf_path: str,
        boxes: list[dict[str, Any]] | None = None,
        output_path: str | None = None,
    ) -> str | None:
        boxes = boxes or self.default_label_boxes()
        try:
            from matplotlib import pyplot as plt  # type: ignore
            from matplotlib.patches import Rectangle  # type: ignore
            from pdf2image import convert_from_path  # type: ignore

            pages = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not pages:
                return None
            image = pages[0]
            width, height = image.size
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(image)
            for box in boxes:
                rect = Rectangle(
                    (box["x"] * width, box["y"] * height),
                    box["w"] * width,
                    box["h"] * height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    box["x"] * width,
                    max(0, box["y"] * height - 5),
                    box["label"],
                    color="red",
                    fontsize=8,
                )
            ax.set_axis_off()
            if output_path:
                plt.savefig(output_path, bbox_inches="tight")
                plt.close(fig)
                return output_path
            plt.show()
            return None
        except Exception as exc:
            raise RuntimeError("Unable to display PDF with boxes. Missing visualization dependencies.") from exc

    def compare_to_rubric(self, analysis: PDFAnalysisResult, rubric: dict[str, Any]) -> dict[str, Any]:
        expected_dims = self._parse_expected_dimensions(rubric.get("dimensions"))
        missing_dims = [d for d in expected_dims if d not in analysis.dimensions_found]
        size_ok = rubric.get("pageSize") is None or (analysis.page_size == str(rubric.get("pageSize")).upper())
        material_expected = rubric.get("material")
        material_ok = (
            material_expected in (None, "", analysis.material)
            or (analysis.material is not None and material_expected is not None and str(material_expected).lower() in analysis.material.lower())
        )

        return {
            "size": {"value": size_ok, "confidence": 5 if size_ok else 2},
            "material": {"value": material_ok, "confidence": 5 if material_ok else 2},
            "scale": {"value": analysis.title_block["value"], "confidence": analysis.title_block["confidence"]},
            "tbComplete": {"value": analysis.title_block["value"], "confidence": analysis.title_block["confidence"]},
            "notes": {"value": analysis.notes["value"], "confidence": analysis.notes["confidence"]},
            "tolerances": {"value": analysis.tolerances["value"], "confidence": analysis.tolerances["confidence"]},
            "projection": {"value": analysis.projection["value"], "confidence": analysis.projection["confidence"]},
            "standard": {"value": analysis.standard["value"], "confidence": analysis.standard["confidence"]},
            "missingDims": {"count": len(missing_dims), "values": missing_dims},
        }

    def _parse_expected_dimensions(self, raw_dimensions: Any) -> list[float]:
        if raw_dimensions is None:
            return []
        if isinstance(raw_dimensions, (int, float)):
            return [float(raw_dimensions)]
        if isinstance(raw_dimensions, list):
            values: list[float] = []
            for item in raw_dimensions:
                values.extend(self._parse_expected_dimensions(item))
            return values
        if isinstance(raw_dimensions, dict):
            values: list[float] = []
            for item in raw_dimensions.values():
                values.extend(self._parse_expected_dimensions(item))
            return values
        if isinstance(raw_dimensions, str):
            return [float(v) for v in re.findall(r"(?:\d+\.\d+|\d+)", raw_dimensions)]
        return []

    def write_results_json(self, results: dict[str, Any], output_json_path: str) -> None:
        Path(output_json_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    def grade(
        self,
        pdf_path: str,
        rubric_json_path: str,
        rubric_key: str,
        output_json_path: str | None = None,
    ) -> dict[str, Any]:
        analysis = self.process_pdf(pdf_path)
        rubric = self.load_rubric(rubric_json_path, rubric_key)
        results = self.compare_to_rubric(analysis, rubric)
        if output_json_path:
            self.write_results_json(results, output_json_path)
        return results
