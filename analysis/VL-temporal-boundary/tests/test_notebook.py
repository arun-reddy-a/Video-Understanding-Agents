import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_code_cells_parse_and_use_committed_manifest():
    notebook = json.loads(
        (PROJECT_ROOT / "video_boundary_analysis_colab.ipynb").read_text(encoding="utf-8")
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
    assert "PROJECT_DIR = REPO_DIR / 'analysis' / 'VL-temporal-boundary'" in code
    assert "os.chdir(PROJECT_DIR)" in code
