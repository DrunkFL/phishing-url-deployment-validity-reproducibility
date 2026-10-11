import csv
import os
from pathlib import Path
import re

import pypandoc


SUBMISSION = Path(__file__).resolve().parent
MARKDOWN_INPUT = SUBMISSION / "main.md"
LATEX_OUTPUT = SUBMISSION / "main.tex"
BIBLIOGRAPHY = SUBMISSION / "references.bib"
CITATION_MAP = SUBMISSION / "citation_key_map.csv"
AUTHOR_BLOCK = """**Dongze Li^1,*^ and Jiahui Liu^1^**  
^1^ School of Computer Science and Technology, Harbin University of Science and Technology, Harbin 150080, China  
*Corresponding author: Dongze Li, ldz0101fl@163.com  
Email: ldz0101fl@163.com (D.L.); liujiahui@hrbust.edu.cn (J.L.)  
ORCID: Dongze Li, 0009-0002-1583-4326

"""


def load_citation_keys() -> dict[int, str]:
    with CITATION_MAP.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        int(row["reference_number"]): row["bibtex_key"]
        for row in rows
        if row["status"] == "EXACT"
    }
    expected = set(range(1, 33))
    if set(mapping) != expected or len(set(mapping.values())) != len(mapping):
        raise AssertionError("Citation map must contain 32 unique exact mappings")
    return mapping


def expand_citation_numbers(specification: str) -> list[int]:
    numbers: list[int] = []
    for token in specification.replace("–", "-").split(","):
        token = token.strip()
        if "-" in token:
            start, stop = (int(value) for value in token.split("-", 1))
            numbers.extend(range(start, stop + 1))
        else:
            numbers.append(int(token))
    return numbers


def main() -> None:
    markdown = MARKDOWN_INPUT.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Expected the manuscript title on the first line")

    title = lines[0][2:].strip()
    body = "\n".join(markdown.splitlines()[1:]).lstrip()
    if not body.startswith(AUTHOR_BLOCK):
        raise ValueError("Expected the verified author metadata block after the title")
    body = body[len(AUTHOR_BLOCK):]

    reference_heading = "\n## References\n"
    if reference_heading not in body:
        raise ValueError("Expected a References section in the Markdown source")
    body, _manual_references = body.split(reference_heading, 1)
    citation_keys = load_citation_keys()

    # Convert the manuscript's compact numeric markers to semantic Pandoc
    # citations.  Natbib then preserves key-level links in the LaTeX source.
    def citation_to_pandoc(match: re.Match[str]) -> str:
        numbers = expand_citation_numbers(match.group(1))
        missing = [number for number in numbers if number not in citation_keys]
        if missing:
            raise AssertionError(f"Unmapped citation numbers: {missing}")
        return "[" + "; ".join(f"@{citation_keys[number]}" for number in numbers) + "]"

    body = re.sub(r"\^\[([0-9][0-9,\-\u2013 ]*)\]\^", citation_to_pandoc, body)

    original_directory = Path.cwd()
    try:
        os.chdir(SUBMISSION)
        pypandoc.convert_text(
            body,
            to="latex",
            format="markdown+citations+tex_math_dollars+tex_math_single_backslash+raw_tex",
            outputfile=str(LATEX_OUTPUT),
            extra_args=[
                "--standalone",
                "--natbib",
                "--bibliography=references.bib",
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
    finally:
        os.chdir(original_directory)

    latex = LATEX_OUTPUT.read_text(encoding="utf-8")
    latex = latex.replace(
        r"\author{}",
        "\\author{Dongze Li\\textsuperscript{1,*} \\and Jiahui Liu\\textsuperscript{1}\\\\"
        "\\small School of Computer Science and Technology, Harbin University of Science and Technology,\\\\"
        "\\small Harbin 150080, China\\\\"
        "\\small Corresponding author: Dongze Li (ldz0101fl@163.com)\\\\"
        "\\small Emails: ldz0101fl@163.com (D.L.); liujiahui@hrbust.edu.cn (J.L.)\\\\"
        "\\small ORCID: Dongze Li (0009-0002-1583-4326)}",
    )

    # Keep the visible numbered superscript style while retaining semantic
    # citation commands and bibliography keys in the source.
    latex = latex.replace(
        r"\usepackage[]{natbib}",
        r"\usepackage[super,sort&compress]{natbib}",
    )
    latex = latex.replace(
        r"\usepackage[super,sort&compress]{natbib}",
        "\\usepackage[super,sort&compress]{natbib}\n\\usepackage[switch]{lineno}",
    )
    latex = latex.replace(r"\maketitle", "\\maketitle\n\\linenumbers", 1)
    latex = latex.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{unsrtnat}")

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
    if "\\cite" not in latex or "\\bibliography{references.bib}" not in latex:
        raise AssertionError("Semantic citations or bibliography binding are missing")
    LATEX_OUTPUT.write_text(latex, encoding="utf-8")

    figure_count = latex.count("\\includegraphics")
    if figure_count != 10:
        raise AssertionError(
            f"Expected 10 embedded main-text figures, found {figure_count}"
        )


if __name__ == "__main__":
    main()
