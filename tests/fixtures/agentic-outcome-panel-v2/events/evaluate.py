"""Held-out checks; executed only inside the bounded evaluation container."""
import importlib.util,json,sys

def load(path):
    spec=importlib.util.spec_from_file_location('candidate_client',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def evaluate(module):
    rows=[]
    def check(name,fn):
        try:fn();rows.append(dict(name=name,passed=True))
        except BaseException as error:rows.append(dict(name=name,passed=False,error=f'{type(error).__name__}: {error}'))
    def run(chunks,expected):
        decoder=module.EventDecoder();actual=[]
        for chunk in chunks:actual.extend(decoder.feed(chunk))
        assert decoder.finish()==[]
        assert actual==expected,(actual,expected)
    event=lambda data,id='',kind='message':dict(event=kind,data=data,id=id)
    fixtures=[
        (b'data: a\n\n',[event('a')]),
        ('id: abc\r\nevent: update\r\ndata: café\r\ndata: 雪\r\n\r\ndata: next\r\n\r\n'.encode(),
            [event('café\n雪','abc','update'),event('next','abc')]),
        (b': comment\nretry: 42\ndata\ndata:  x \nunknown: ignored\n\n',[event('\n x ')]),
        (b'id: first\n\nid: bad\0id\ndata: one\n\nid:\ndata: two\n\n',[event('one','first'),event('two')]),
        (b'event: unused\n\ndata: ordinary\n\nevent: x\nevent:\ndata:\n\n',[event('ordinary'),event('')]),
        (b'data: a\r\rdata: b\ndata: c\r\n\r\n',[event('a'),event('b\nc')]),
        (b'Data: ignored\n: x\n\n',[]),
        (b'data: incomplete\n',[]),
        (b'data: incomplete',[]),
    ]
    for index,(wire,expected)in enumerate(fixtures):
        check(f'whole_{index}',lambda wire=wire,expected=expected:run([wire],expected))
        check(f'byte_chunks_{index}',lambda wire=wire,expected=expected:run([bytes([b])for b in wire],expected))
        def every_split(wire=wire,expected=expected):
            for split in range(len(wire)+1):run([wire[:split],b'',wire[split:]],expected)
        check(f'every_two_chunk_split_{index}',every_split)
    def no_early_dispatch():
        d=module.EventDecoder();assert d.feed(b'data: held\n')==[]
        assert d.feed(b'')==[];assert d.feed(b'\n')==[event('held')]
    check('dispatch_only_on_blank_line',no_early_dispatch)
    def eof_closed():
        d=module.EventDecoder();assert d.feed(b'data: discard\n')==[]
        assert d.finish()==[] and d.finish()==[]
        try:d.feed(b'')
        except RuntimeError:return
        raise AssertionError('feed after finish accepted')
    check('finish_discards_pending_and_closes',eof_closed)
    def truncated_utf8():
        d=module.EventDecoder();assert d.feed(b'data: \xe2\x82')==[]
        try:d.finish()
        except UnicodeDecodeError:return
        raise AssertionError('truncated UTF8 accepted')
    check('strict_utf8_at_eof',truncated_utf8)
    def invalid_utf8():
        try:module.EventDecoder().feed(b'data: \xff\n\n')
        except UnicodeDecodeError:return
        raise AssertionError('invalid UTF8 accepted')
    check('strict_utf8_during_feed',invalid_utf8)
    def reject_text():
        try:module.EventDecoder().feed('data: x\n\n')
        except TypeError:return
        raise AssertionError('text feed accepted')
    check('bytes_api',reject_text)
    return dict(task='incremental-event-decoder-v1',tests=rows,passed_required=sum(r['passed']for r in rows),
        required_checks=len(rows),task_checks_pass=all(r['passed']for r in rows))

if __name__=='__main__':
    try:result=evaluate(load(sys.argv[1]))
    except BaseException as error:result=dict(task_checks_pass=False,grader_error=f'{type(error).__name__}: {error}')
    print('CANADA_OUTCOME='+json.dumps(result))
