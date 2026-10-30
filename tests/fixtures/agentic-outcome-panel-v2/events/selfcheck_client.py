"""Known-good control for the explicit fixture contract; never shown to the model."""
import codecs


class EventDecoder:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.closed = False
        self.line = ''
        self.after_cr = False
        self.data = []
        self.event = ''
        self.last_id = ''

    def _line(self, line):
        if line == '':
            result = []
            if self.data:
                result.append({'event': self.event or 'message',
                    'data': '\n'.join(self.data), 'id': self.last_id})
            self.data = []
            self.event = ''
            return result
        if line.startswith(':'):
            return []
        field, separator, value = line.partition(':')
        if not separator:
            value = ''
        elif value.startswith(' '):
            value = value[1:]
        if field == 'data':
            self.data.append(value)
        elif field == 'event':
            self.event = value
        elif field == 'id' and '\0' not in value:
            self.last_id = value
        return []

    def feed(self, chunk):
        if self.closed:
            raise RuntimeError('decoder closed')
        if not isinstance(chunk, bytes):
            raise TypeError('bytes required')
        events = []
        for char in self.decoder.decode(chunk):
            if self.after_cr:
                self.after_cr = False
                if char == '\n':
                    continue
            if char in '\r\n':
                events.extend(self._line(self.line))
                self.line = ''
                self.after_cr = char == '\r'
            else:
                self.line += char
        return events

    def finish(self):
        if self.closed:
            return []
        self.closed = True
        self.decoder.decode(b'', final=True)
        self.line = ''
        self.data = []
        return []
