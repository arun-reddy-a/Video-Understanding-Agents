# Analysis 3: Overleaf integration

The complete contribution is in `analysis3.tex`: four subsections with each PNG figure placed immediately after its explanatory paragraph. Every figure reference resolves within this file. The prose remains approximately 474 words; including all four figures will likely exceed one ICML two-column page.

## Upload and insert

1. Upload `analysis3.tex` and the four PNGs in `figures/` to the Overleaf project root, preserving the `figures/` directory. Alternatively, upload this whole directory and prefix each `\input` and `\includegraphics` path with its uploaded folder name.
2. Ensure the main preamble loads `\usepackage{graphicx}`. Use the team's existing ICML 2026 template and bibliography settings.
3. Replace the entire `\section{Analysis 3: Descriptive Title}` placeholder and its `\dummytext` with:

```latex
\input{analysis3}
```

Compile twice to resolve figure references. Figure numbers are assigned by the main report; the supplied labels are unique and do not assume fixed numbers. No extra bibliography entries are required by this section. Retain the team's existing citations for Video-MME and reader models where those are introduced.

## Figure placement and page-length check

Replace only the Analysis 3 placeholder with `analysis3.tex`, or use `\input{analysis3}` there. No changes to the rest of the report or its preamble are needed: your existing `graphicx` package is sufficient.

Each image uses a standard single-column `figure[!htbp]` immediately after its explanatory paragraph. LaTeX can place it here or at a nearby column top, bottom, or float page, allowing text to fill the remaining space. This avoids large blank areas caused by unbreakable inline image blocks. No additional packages or global formatting changes are needed.

No experimental code or GPU is needed: existing PNGs are reused unchanged. The workspace lacks the ICML style file and a LaTeX compiler, so check the final layout in Overleaf. All four figures may require more than one page; forcing them into one page can make their panel labels too small.

## Source and verification

Figures were copied unchanged from `../analysis_assignment_2/outputs/`. Reported statistics were checked against `a1_language/summary_by_task.csv`, `a2_signal/summary.csv`, and the template-option selector, overlap, and confidence-interval CSVs in `a3_task/`. Scoring and voting descriptions follow `common.py`, `a2_signal.py`, and `a3_task.py`.

The main text qualifies the approximate null threshold, question-level bootstrap intervals, and frame-vote proxy. Planned changes and the Qwen2.5-VL evaluation are explicitly future work. No experiments were rerun.

Assignment 2 also requires a separate AI-use and teammate-contribution disclosure page. Record this AI assistance with interpreting saved code/results and drafting LaTeX, together with the team's other actual AI uses and contributions. Keep the project repository link in the main report.
