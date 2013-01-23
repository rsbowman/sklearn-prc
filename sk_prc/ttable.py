
class TextTable(object):
    def __init__(self, headers, formats):
        self._headers = headers
        self._widths = map(len, headers)
        self._formats = formats
                
    def header(self):
        return ' '.join(self._headers)

    def row(self, *args):
        assert len(args) == len(self._widths), \
            "number of rows doesn't match number of headers"
        cells = []
        for a, w, f in zip(args, self._widths, self._formats):
            if f.startswith(' '):
                sign = ' '
            else:
                sign = ''
            mv = '{{:<{}{}{}}}'.format(sign, w, f.strip())
            try:
                cells.append(mv.format(a))
            except ValueError:
                print '-->', mv
                raise
        return ' '.join(cells)

