"""Load the documents in a folder and cut them into chunks (notebook section 6)."""
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import (
    CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schemas import DocumentInfo

log = logging.getLogger(__name__)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": lambda p: CSVLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".txt": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".md": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
}


def load_documents(folder: Path) -> Tuple[List[Document], List[str]]:
    """The documents with text, and the names of the files nothing could be read from."""
    docs, unreadable = [], []
    for path in sorted(folder.rglob("*")):
        make_loader = LOADERS.get(path.suffix.lower())
        if make_loader is None or not path.is_file():
            continue
        try:
            loaded = [d for d in make_loader(str(path)).load() if d.page_content.strip()]
        except Exception as e:
            log.warning("Skipped %s: %s", path.name, e)
            unreadable.append(path.name)
            continue
        if not loaded:
            log.warning("No text extracted from %s (scanned PDF?)", path.name)
            unreadable.append(path.name)
        docs.extend(loaded)
    return docs, unreadable


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return filter_complex_metadata(splitter.split_documents(docs))


def file_report(docs: List[Document], unreadable: List[str] = ()) -> List[DocumentInfo]:
    """Per-file report. Very little text per page usually means the PDF is made of
    images (scans or screenshots): that content cannot be searched without OCR."""
    per_file = {}
    for d in docs:
        name = Path(str(d.metadata.get("source", "?"))).name
        pages, chars = per_file.get(name, (0, 0))
        per_file[name] = (pages + 1, chars + len(d.page_content))
    return [
        DocumentInfo(name=name, pages=pages, characters=chars, low_text=chars / pages < 200)
        for name, (pages, chars) in per_file.items()
    ] + [DocumentInfo(name=name, pages=0, characters=0, low_text=True) for name in unreadable]
