from nebula3.data.DataObject import ValueWrapper

class VidCodec:
    """Detect Nebula VID type and format literals accordingly."""
    def __init__(self, is_string: bool, raw_type: str = ""):
        self.is_string = is_string
        self.raw_type = raw_type
        self._cache = {}

    @staticmethod
    def detect(session, space: str) -> "VidCodec":
        resp = session.execute(f"DESCRIBE SPACE `{space}`")
        if not resp.is_succeeded() or not resp.rows():
            return VidCodec(True, "UNKNOWN")
        cols = [c.decode() if isinstance(c, (bytes, bytearray)) else c for c in resp.keys()]
        try:
            idx = cols.index("Vid Type")
        except ValueError:
            return VidCodec(True, "UNKNOWN")
        s = str(ValueWrapper(resp.rows()[0].values[idx]).cast()).strip('"').upper()
        return VidCodec("FIXED_STRING" in s, s)

    @staticmethod
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("\"", "\\\"")

    def literal(self, v) -> str:
        if v in self._cache:
            return self._cache[v]
        if self.is_string:
            lit = f"\"{self._esc(str(v))}\""
        else:
            try:
                lit = str(int(v))
            except Exception as e:
                raise TypeError(f"Space VID=INT64, but the input VID cannot be converted to an integer: {v!r}") from e
        self._cache[v] = lit
        return lit