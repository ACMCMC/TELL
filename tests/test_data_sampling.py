from types import SimpleNamespace

from rl_detector import data


def test_clean_document_text_removes_html_tags_preserves_entity_text():
    text = "Hello<br><br><p>world&nbsp;&amp; more</p>\n<div>again</div>"

    assert data.clean_document_text(text) == "Hello world&nbsp;&amp; more again"


def test_manifest_docs_are_cleaned(tmp_path, monkeypatch):
    manifest = tmp_path / "docs.jsonl"
    manifest.write_text('{"text": "A<br> B&nbsp;&amp; C", "label": 1}\n', encoding="utf-8")
    monkeypatch.setattr(
        data,
        "CFG",
        SimpleNamespace(
            data=SimpleNamespace(train_docs_path=str(manifest)),
            frozen=SimpleNamespace(seed=123),
        ),
    )

    docs = data.load_docs(["unused"], max_docs=1)

    assert docs == [{"text": "A B&nbsp;&amp; C", "label": 1}]


def test_metadata_is_preserved_from_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "docs.jsonl"
    manifest.write_text(
        '{"text": "A", "label": 1, "doc_type": "forum_qa", "source": "reddit"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        data,
        "CFG",
        SimpleNamespace(
            data=SimpleNamespace(train_docs_path=str(manifest)),
            frozen=SimpleNamespace(seed=123),
        ),
    )

    docs = data.load_docs(["unused"], max_docs=1)

    assert docs == [{"text": "A", "label": 1, "doc_type": "forum_qa", "source": "reddit"}]

# old monkeypatched multi-dataset HF row tests removed; load_docs is cfg manifest path only now
