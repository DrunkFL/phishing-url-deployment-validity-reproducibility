from pathlib import Path
import re

import pypandoc


SUBMISSION = Path(__file__).resolve().parent
MARKDOWN_INPUT = SUBMISSION / "main.md"
LATEX_OUTPUT = SUBMISSION / "main.tex"


def main() -> None:
    markdown = MARKDOWN_INPUT.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Expected the manuscript title on the first line")

    title = lines[0][2:].strip()
    body = "\n".join(markdown.splitlines()[1:]).lstrip()

    # The manuscript uses compact citation markers such as ^[1-10]^ in its
    # Markdown source.  Pandoc otherwise treats these as footnotes, whereas the
    # journal requires consecutive Arabic superscript citations.
    def citation_to_latex(match: re.Match[str]) -> str:
        citation_numbers = match.group(1).replace("-", "--").replace("–", "--")
        return rf"\textsuperscript{{{citation_numbers}}}"

    body = re.sub(r"\^\[([0-9][0-9,\-\u2013 ]*)\]\^", citation_to_latex, body)

    pypandoc.convert_text(
        body,
        to="latex",
        format="markdown+tex_math_dollars+tex_math_single_backslash+raw_tex",
        outputfile=str(LATEX_OUTPUT),
        extra_args=[
            "--standalone",
            "--shift-heading-level-by=-1",
            "--resource-path",
            str(SUBMISSION),
            "--metadata",
            f"title={title}",
            "--variable",
            "documentclass=article",
            "--variable",
            "geometry:margin=1in",
            "--variable",
            "fontsize=10pt",
        ],
    )

    latex = LATEX_OUTPUT.read_text(encoding="utf-8")

    # Markdown captions already contain their figure number.  The LaTeX
    # figure counter supplies it again, so remove only the duplicated prefix.
    latex = re.sub(r"\\caption\{Figure \d+\.\s*", r"\\caption{", latex)

    # Bare URLs are readable in Markdown but cannot line-break safely in TeX.
    # Wrap them with \url while preserving sentence punctuation.
    def wrap_url(match: re.Match[str]) -> str:
        token = match.group(0)
        trailing = ""
        while token and token[-1] in ".,;":
            trailing = token[-1] + trailing
            token = token[:-1]
        token = token.replace(r"\_", "_")
        return rf"\url{{{token}}}{trailing}"

    latex = re.sub(r"(?<!\\url\{)https://[^\s}]+", wrap_url, latex)

    # Commit identifiers and SHA-256 values are emitted by Pandoc as
    # unbreakable \texttt spans.  Treat long hexadecimal identifiers like
    # URLs so xurl can wrap them without changing their visible value.
    latex = re.sub(
        r"\\texttt\{([0-9a-fA-F]{40,64})\}",
        r"\\nolinkurl{\1}",
        latex,
    )

    if "\\footnote{" in latex:
        raise AssertionError("Citation conversion left footnotes in the LaTeX output")
    LATEX_OUTPUT.write_text(latex, encoding="utf-8")

    figure_count = latex.count("\\includegraphics")
    if figure_count != 7:
        raise AssertionError(
            f"Expected 7 embedded main-text figures, found {figure_count}"
        )


if __name__ == "__main__":
    main()
