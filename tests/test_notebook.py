import ast
import json
from pathlib import Path


def test_notebook_code_cells_parse_and_use_committed_manifest():
    notebook = json.loads(
        Path("video_boundary_analysis_colab.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "manifests' / 'long_stratified_24.csv" in code
    assert "results['events']" in code
    assert "_similarities.png" in code
    assert "arun-reddy-a/Video-Understanding-Agents" in code
    assert "BRANCH = 'VL-temporal-boundries'" in code
    assert "'--branch', BRANCH" in code
