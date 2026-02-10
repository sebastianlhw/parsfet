
import gzip
import pytest
from pathlib import Path
from parsfet.parsers.base import BaseParser

class DummyParser(BaseParser):
    def parse(self, path):
        return "parsed"
    
    def parse_string(self, content, name="unknown"):
        return content

def test_read_file_plain(tmp_path):
    parser = DummyParser()
    f = tmp_path / "test.txt"
    f.write_text("hello world", encoding="utf-8")
    
    assert parser._read_file(f) == "hello world"

def test_read_file_gzip(tmp_path):
    parser = DummyParser()
    f = tmp_path / "test.txt.gz"
    with gzip.open(f, "wt", encoding="utf-8") as gf:
        gf.write("hello gzip")
        
    assert parser._read_file(f) == "hello gzip"

def test_tokenizer_methods():
    parser = DummyParser()
    tokens = ["A", "B", "C"]
    parser._init_tokens(tokens)
    
    # Peek
    assert parser._peek() == "A"
    assert parser._peek(1) == "B"
    assert parser._peek(2) == "C"
    assert parser._peek(3) is None
    
    # Consume
    assert parser._consume() == "A"
    assert parser._pos == 1
    
    # Expect success
    assert parser._expect("B") == "B"
    
    # Expect failure
    with pytest.raises(ValueError, match="Expected 'X', got 'C'"):
        parser._expect("X")
        
    # Expect case insensitive
    parser._pos -= 1 # Rewind to C
    assert parser._expect("c", case_sensitive=False) == "C"
    
    # End of stream
    assert parser._consume() is None

def test_skip_semicolon():
    parser = DummyParser()
    parser._init_tokens([";", "A"])
    
    parser._skip_semicolon()
    assert parser._peek() == "A"
    
    parser._skip_semicolon() # Should do nothing
    assert parser._peek() == "A"
