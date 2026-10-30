from client import EventDecoder


def test_complete_message():
    assert EventDecoder().feed(b'data: hello\n\n') == [
        {'event': 'message', 'data': 'hello', 'id': ''}]


def test_waits_for_blank_line():
    decoder = EventDecoder()
    assert decoder.feed(b'data: hello\n') == []
    assert decoder.feed(b'\n') == [{'event': 'message', 'data': 'hello', 'id': ''}]
