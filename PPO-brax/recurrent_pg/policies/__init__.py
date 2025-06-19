from recurrent_pg.policies.lstm import LSTM

from recurrent_pg.policies.sg_rsnn_continuous import RSNN


POLICIES = {
    "LSTM": LSTM,

    "RSNN": RSNN
}
