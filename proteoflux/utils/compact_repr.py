import reprlib

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


class CompactRepr(reprlib.Repr):
    def __init__(self, max_items: int = 10):
        super().__init__()
        self.maxlist = max_items
        self.maxtuple = max_items
        self.maxset = max_items
        self.maxdict = max_items
        self.maxstring = 200

    def repr_polars_df(self, obj):
        return f"polars.DataFrame(shape={obj.shape})\n{obj.head(10)}"

    def repr_pandas_df(self, obj):
        return f"pandas.DataFrame(shape={obj.shape})\n{obj.head(10)}"

    def repr(self, obj):
        if pl is not None and isinstance(obj, pl.DataFrame):
            return self.repr_polars_df(obj)
        if pd is not None and isinstance(obj, pd.DataFrame):
            return self.repr_pandas_df(obj)
        return super().repr(obj)


compact_repr = CompactRepr()

