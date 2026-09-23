from arxiv_agent import pdf_utils

FAKE_PAGE_1 = """
Abstract
This paper studies foo.

Introduction
Foo is important because bar.
"""

FAKE_PAGE_2 = """
Method
We do X then Y.

Results
X improves Y by 10 percent.
"""

FAKE_PAGE_3 = """
Limitations
This only works on toy data.

References
[1] Someone et al 2020.
[2] Someone else et al 2021.
"""


def test_split_sections_with_pages_assigns_page_labels():
    pages = [FAKE_PAGE_1, FAKE_PAGE_2, FAKE_PAGE_3]
    sections, page_sections = pdf_utils._split_sections_with_pages(pages)

    assert "method" in sections and "results" in sections and "limitations" in sections
    assert len(page_sections) == 3
    # page 1 ends inside "introduction"
    assert page_sections[0] == "introduction"
    # page 2 ends inside "results"
    assert page_sections[1] == "results"
    # page 3 ends inside "references"
    assert page_sections[2] == "references"


def test_extract_references_parses_numbered_list():
    sections, _ = pdf_utils._split_sections_with_pages([FAKE_PAGE_1, FAKE_PAGE_2, FAKE_PAGE_3])
    refs = pdf_utils._extract_references(sections)
    assert len(refs) == 2
    assert refs[0].startswith("[1]")


def test_extract_references_empty_when_no_references_section():
    sections, _ = pdf_utils._split_sections_with_pages([FAKE_PAGE_1])
    assert pdf_utils._extract_references(sections) == []


def test_backward_compatible_split_sections_flat_string():
    flat = FAKE_PAGE_1 + FAKE_PAGE_2 + FAKE_PAGE_3
    sections = pdf_utils._split_sections(flat)
    assert "method" in sections and "limitations" in sections
