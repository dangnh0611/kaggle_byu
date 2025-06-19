import logging
import os

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class SubmissionDataFrame:

    def __init__(self):
        self.all_x = []
        self.all_y = []
        self.all_z = []
        self.all_conf = []
        self.all_tomo_id = []

    def clear(self):
        self.all_x.clear()
        self.all_y.clear()
        self.all_z.clear()
        self.all_tomo_id.clear()

    def add_row(self, tomo_id, x, y, z, conf):
        self.all_tomo_id.append(tomo_id)
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_z.append(z)
        self.all_conf.append(conf)

    def to_polars(self, submit=False):
        assert len(self.all_tomo_id) == len(set(self.all_tomo_id))
        d = {
            "tomo_id": pl.Series(self.all_tomo_id),
            "motor_z": pl.Series(self.all_z),
            "motor_y": pl.Series(self.all_y),
            "motor_x": pl.Series(self.all_x),
        }
        if not submit:
            d["conf"] = pl.Series(self.all_conf)
        df = pl.DataFrame(d)
        return df

    def to_pandas(self, submit=False):
        # assert len(self.all_tomo_id) == len(set(self.all_tomo_id))
        d = {
            "tomo_id": self.all_tomo_id,
            "motor_z": self.all_z,
            "motor_y": self.all_y,
            "motor_x": self.all_x,
        }
        if not submit:
            d["conf"] = self.all_conf
        df = pd.DataFrame(d)
        return df

    def write_csv(self, fname, submit=False):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df = self.to_polars(submit=submit)
        df.write_csv(fname)
