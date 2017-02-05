from . import crossword_extractor

def test_crossword_extractor():
    assert crossword_extractor.apply("Jane") == "hello Jane"
