class EventDecoder:
    def feed(self, chunk):
        events = []
        for line in chunk.decode('utf-8').splitlines():
            if line.startswith('data:'):
                events.append({'event': 'message', 'data': line[5:].strip(), 'id': ''})
        return events

    def finish(self):
        return []
