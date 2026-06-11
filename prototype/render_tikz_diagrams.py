#!/usr/bin/env python3
"""Render the PACER paper's TikZ line-art figures as vector PDFs."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def _latex_engine() -> list[str]:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is not None:
        return [pdflatex, "-interaction=nonstopmode", "-halt-on-error"]

    tectonic = ROOT / ".codex_tools" / "tectonic.exe"
    if tectonic.exists():
        return [str(tectonic), "--keep-logs", "--keep-intermediates"]

    raise RuntimeError(
        "neither pdflatex nor the bundled Tectonic executable is available "
        "for TikZ figure rendering"
    )


def _source(filename: str) -> str:
    return (FIGURES / filename).read_text(encoding="utf-8")


def _compile_tikz(name: str, tex: str, out_pdf: Path) -> None:
    out_pdf = out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.with_name(out_pdf.stem + "_source.tex").write_text(tex, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix=f"pacer_{name}_") as td:
        work = Path(td)
        src = work / f"{name}.tex"
        src.write_text(tex, encoding="utf-8")

        proc = subprocess.run(
            [*_latex_engine(), src.name],
            cwd=work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            log_path = out_pdf.with_suffix(".tikz.log")
            log_path.write_text(proc.stdout, encoding="utf-8")
            raise RuntimeError(f"TikZ render failed for {name}; see {log_path}")

        shutil.copyfile(work / f"{name}.pdf", out_pdf)


def render_architecture(out_pdf: Path) -> None:
    _compile_tikz(
        "fig_semantics_architecture",
        _source("fig_semantics_architecture_source.tex"),
        out_pdf,
    )


def render_executor(out_pdf: Path) -> None:
    _compile_tikz(
        "fig_executor_contract",
        _source("fig_executor_contract_source.tex"),
        out_pdf,
    )


if __name__ == "__main__":
    render_architecture(FIGURES / "fig_semantics_architecture.pdf")
    render_executor(FIGURES / "fig_executor_contract.pdf")
