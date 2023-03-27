"""Test basic parser implementation"""

from functools import reduce
import json
from typing import Literal

from pydantic import BaseModel

from clownfish import StringParser, ListParser, DictParser, Candidates, LiteralParser, UnionParser, NumberParser

TEST_SCHEMA = json.loads("""
        {
        "title": "Drawing",
        "type": "object",
        "properties": {
            "shapes": {
            "title": "Shapes",
            "type": "array",
            "items": {
                "type": "string"
            }
            }
        },
        "required": [
            "shapes"
        ]
        }""")

def test_string():
    """Test that string parsing works for good strings and not for bad ones"""

    string_parser = StringParser.init()
    assert reduce(string_parser.step, '"ok"', string_parser) is not None
    assert reduce(string_parser.step, ' bad', string_parser) is None

def test_list():
    """Test that list parsing works"""

    list_parser = ListParser.init(TEST_SCHEMA['properties']['shapes'], TEST_SCHEMA)
    assert reduce(list_parser.step, '[]', list_parser) is not None
    assert reduce(list_parser.step, '[""]', list_parser) is not None
    assert reduce(list_parser.step, '["foo","bar"]', list_parser) is not None
    assert reduce(list_parser.step, 'bad', list_parser) is None
    assert reduce(list_parser.step, '[,,]', list_parser) is None

def test_dict():
    """Test that dict parsing works"""

    dict_parser = DictParser.init(TEST_SCHEMA, TEST_SCHEMA)
    assert reduce(dict_parser.step, '{"shapes":[]}', dict_parser) is not None
    assert reduce(dict_parser.step, '{"shapes": ["circle"]}', dict_parser) is not None
    assert reduce(dict_parser.step, '{"shapes2":[]}', dict_parser) is None
    assert reduce(dict_parser.step, '{"shapes":[], "nope": 9}', dict_parser) is None
    assert reduce(dict_parser.step, '{"shapes2":[}', dict_parser) is None

def test_literal():
    """Test that literal 'parsing' works"""

    literal_parser = LiteralParser.init('"foobar"')
    assert reduce(literal_parser.step, '"foobar"', literal_parser) is not None
    assert reduce(literal_parser.step, '"foo', literal_parser) is not None
    assert reduce(literal_parser.step, '"baz"', literal_parser) is None

def test_union_enum():
    """Test that simple unions of value"""

    union_parser = UnionParser.init([LiteralParser.init(json.dumps(v)) for v in ['reboot', 'recharge']], json.loads('{}'))
    assert reduce(union_parser.step, '"', union_parser) is not None
    assert reduce(union_parser.step, '"re', union_parser) is not None
    assert reduce(union_parser.step, '"red', union_parser) is None
    assert reduce(union_parser.step, '"reboot"', union_parser) is not None
    assert reduce(union_parser.step, '"recharge"', union_parser) is not None

def test_number():
    number_parser = NumberParser.init()
    assert reduce(number_parser.step, '88', number_parser) is not None
    assert reduce(number_parser.step, '88.8', number_parser) is not None
    assert reduce(number_parser.step, '88..8', number_parser) is None
    assert reduce(number_parser.step, '"ok"', number_parser) is None

def test_number_list():
    schema = json.loads("""
        {
        "title": "Drawing",
        "type": "object",
        "properties": {
            "shapes": {
            "title": "Shapes",
            "type": "array",
            "items": {
                "type": "number"
            }
            }
        },
        "required": [
            "shapes"
        ]
        }""")

    list_parser = ListParser.init(schema['properties']['shapes'], schema)
    assert reduce(list_parser.step, '[8]', list_parser) is not None
    assert reduce(list_parser.step, '[888,866]', list_parser) is not None
    assert reduce(list_parser.step, '[8..,]', list_parser) is None
    assert reduce(list_parser.step, '[8,]', list_parser) is None
    assert reduce(list_parser.step, '[8,', list_parser) is not None

    assert ',' in list_parser.candidates(reduce(list_parser.step, '[8', list_parser))
    assert ']' in list_parser.candidates(reduce(list_parser.step, '[8', list_parser))
    assert '6' in list_parser.candidates(reduce(list_parser.step, '[8', list_parser))
    assert '6' in list_parser.candidates(reduce(list_parser.step, '[8,', list_parser))
    assert ',' not in list_parser.candidates(reduce(list_parser.step, '[8,', list_parser))

def test_number_dict():
    schema = json.loads("""
        {
        "title": "Drawing",
        "type": "object",
        "properties": {
            "shapes": {
                "type": "number"
            },
            "lines": {
                "type": "string"
            }
        },
        "required": [
            "shapes"
        ]
        }""")
    dict_parser = DictParser.init(schema, schema)
    assert reduce(dict_parser.step, '{"shapes":7', dict_parser) is not None
    assert reduce(dict_parser.step, '{"shapes":77', dict_parser) is not None
    assert reduce(dict_parser.step, '{"shapes":77,"lines":"1",', dict_parser) is None
    assert reduce(dict_parser.step, '{"shapes":77}', dict_parser) is not None

    assert ', ' in dict_parser.candidates(reduce(dict_parser.step, '{"shapes":7', dict_parser))
    assert '7' in dict_parser.candidates(reduce(dict_parser.step, '{"shapes":7', dict_parser))
    assert '7' in dict_parser.candidates(reduce(dict_parser.step, '{"shapes":77', dict_parser))
    assert '7' not in dict_parser.candidates(reduce(dict_parser.step, '{"shapes":77,', dict_parser))
    assert ', ' not in dict_parser.candidates(reduce(dict_parser.step, '{"lines":"2", "shapes":77', dict_parser))

def test_string_candidates():
    """Tests that a string parser generates coherenet candidates"""

    string_parser = StringParser.init()
    assert StringParser.candidates(reduce(string_parser.step, '"ok"', string_parser)) == Candidates.NONE
    assert StringParser.candidates(reduce(string_parser.step, '', string_parser)) == ['"']
    assert StringParser.candidates(reduce(string_parser.step, '"o', string_parser)) == Candidates.ANY

def test_list_candidates():
    """Tests that list parsing generates coherent candidates"""

    list_parser = ListParser.init(TEST_SCHEMA['properties']['shapes'], TEST_SCHEMA)
    assert ListParser.candidates(reduce(list_parser.step, '[]', list_parser)) == Candidates.NONE
    assert ListParser.candidates(reduce(list_parser.step, '', list_parser)) == ['[']
    assert ListParser.candidates(reduce(list_parser.step, '["foo"', list_parser)) == [',', ']']
    assert ListParser.candidates(reduce(list_parser.step, '[', list_parser)) == ['"', ']']

def test_dict_candidates():
    """Tests that dict parsing generates coherent candidates"""

    dict_parser = DictParser.init(TEST_SCHEMA, TEST_SCHEMA)
    assert DictParser.candidates(reduce(dict_parser.step, '{', dict_parser)) == ['"shapes": ']
    assert DictParser.candidates(reduce(dict_parser.step, '{"', dict_parser)) == ['shapes":']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shap', dict_parser)) == ['es":']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shapes"', dict_parser)) == [': ']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shapes":', dict_parser)) == ['[']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shapes":', dict_parser)) == ['[']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shapes":[]', dict_parser)) == [' }']
    assert DictParser.candidates(reduce(dict_parser.step, '{"shapes":[]}', dict_parser)) == Candidates.NONE

def test_literal_candidates():
    """Tests that literal parsing generates coherent candidates"""

    literal_parser = LiteralParser.init('"foobar"')
    assert literal_parser.candidates(reduce(literal_parser.step, '', literal_parser)) == ['"foobar"']
    assert literal_parser.candidates(reduce(literal_parser.step, '"foo', literal_parser)) == ['bar"']
    assert literal_parser.candidates(reduce(literal_parser.step, '"foobar"', literal_parser)) == Candidates.NONE

def test_union_enum_candidates():
    """Test that simple unions of value"""

    union_parser = UnionParser.init([LiteralParser.init(json.dumps(v)) for v in ['reboot', 'recharge']], json.loads('{}'))
    assert union_parser.candidates(reduce(union_parser.step, '', union_parser)) == ['"reboot"', '"recharge"']
    assert union_parser.candidates(reduce(union_parser.step, '"re', union_parser)) == ['boot"', 'charge"']
    assert union_parser.candidates(reduce(union_parser.step, '"reb', union_parser)) == ['oot"']
    assert union_parser.candidates(reduce(union_parser.step, '"reboot"', union_parser)) == Candidates.NONE


def test_with_pydantic():
    class Note(BaseModel):
        note: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
        beat_idx: float

    class Song(BaseModel):
        notes: list[Note]

    schema = json.loads(Song.schema_json())
    print(schema)

    dict_parser = DictParser.init(schema, schema)

    assert reduce(dict_parser.step, '{"notes":[]}', dict_parser) is not None
    assert reduce(dict_parser.step, '{"notes":[{"note":"A","beat_idx":7}]}', dict_parser) is not None
    assert '"note": ' in dict_parser.candidates(reduce(dict_parser.step, '{"notes":[{', dict_parser))
    assert '"beat_idx": ' in dict_parser.candidates(reduce(dict_parser.step, '{"notes":[{', dict_parser))