"""OTSL → HTML tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swissdocai.otsl import convert_otsl_tables  # noqa: E402
from swissdocai.vlm import clean_output  # noqa: E402


def test_simple_table_inside_markdown():
    md = "Text before.\n\n<fcel>Metric<fcel>SA<nl><fcel>Latency<fcel>142.5<nl>\n\n## Next"
    assert convert_otsl_tables(md) == (
        "Text before.\n\n<table>\n<tr><td>Metric</td><td>SA</td></tr>\n<tr><td>Latency</td><td>142.5</td></tr>\n"
        "</table>\n\n## Next")


def test_spans_headers_and_empty_cells():
    otsl = "<otsl><ched>Name<ched>Betrag<lcel><nl><fcel>Müller<fcel>10<fcel>CHF<nl><ucel><ecel><fcel>EUR<nl></otsl>"
    assert convert_otsl_tables(otsl) == (
        "<table>\n<tr><th>Name</th><th colspan=\"2\">Betrag</th></tr>\n"
        "<tr><td rowspan=\"2\">Müller</td><td>10</td><td>CHF</td></tr>\n<tr><td></td><td>EUR</td></tr>\n</table>")


def test_cell_text_is_escaped_and_plain_markdown_untouched():
    assert "<td>a &lt; b</td>" in convert_otsl_tables("<fcel>a < b<nl>")
    assert convert_otsl_tables("No tables here <b>bold</b>") == "No tables here <b>bold</b>"


def test_generated_output_gets_html_tables():
    assert clean_output("```markdown\n<fcel>A<fcel>B<nl>\n```").startswith("<table>")
