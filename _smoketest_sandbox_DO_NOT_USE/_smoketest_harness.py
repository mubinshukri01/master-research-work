"""Offline smoke-test harness for the three V3 scripts.

Not part of the deliverable — it exists only to prove the scripts execute
end to end (shapes, serialisation, checkpoint round-trip, Excel export)
without a GPU, without TensorFlow, and without downloading BERT weights.

It fakes:
  * tensorflow.keras.preprocessing.{text,sequence}  -> pure-python equivalents
  * BertTokenizerFast / BertModel.from_pretrained   -> tiny randomly-init BERT
and shrinks the search budgets so a full run takes seconds.
"""
import json
import os
import sys
import types
import numpy as np

# The sandbox has no CUDA libraries; load torch with lazy symbol binding so the
# CPU-only build imports against stub .so files. Harmless on a real machine.
if os.environ.get("SMOKETEST_LAZY_DLOPEN"):
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

ROOT = os.path.dirname(os.path.abspath(__file__))
# Run in a scratch directory so the test never writes into the deliverable folder.
WORKDIR = os.environ.get("SMOKETEST_WORKDIR", "/tmp/smoketest_run")
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

# ---------------------------------------------------------------
# Fake tensorflow.keras.preprocessing
# ---------------------------------------------------------------
class FakeTokenizer:
    def __init__(self, num_words=None, oov_token=None, **kw):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(str(t).lower().split())
        words = [w for w, _ in c.most_common()]
        if self.oov_token:
            words = [self.oov_token] + words
        self.word_index = {w: i + 1 for i, w in enumerate(words)}
        self.index_word = {i: w for w, i in self.word_index.items()}

    def texts_to_sequences(self, texts):
        oov = self.word_index.get(self.oov_token)
        out = []
        for t in texts:
            seq = []
            for w in str(t).lower().split():
                i = self.word_index.get(w)
                if i is None or (self.num_words and i >= self.num_words):
                    if oov is not None:
                        seq.append(oov)
                else:
                    seq.append(i)
            out.append(seq)
        return out

    def to_json(self):
        return json.dumps({"num_words": self.num_words, "oov_token": self.oov_token,
                           "word_index": self.word_index})


def fake_tokenizer_from_json(s):
    d = json.loads(s)
    t = FakeTokenizer(num_words=d["num_words"], oov_token=d["oov_token"])
    t.word_index = d["word_index"]
    t.index_word = {i: w for w, i in t.word_index.items()}
    return t


def fake_pad_sequences(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:]
        if s:
            out[i, maxlen - len(s):] = s
    return out


tf = types.ModuleType("tensorflow")
keras = types.ModuleType("tensorflow.keras")
prep = types.ModuleType("tensorflow.keras.preprocessing")
text_mod = types.ModuleType("tensorflow.keras.preprocessing.text")
seq_mod = types.ModuleType("tensorflow.keras.preprocessing.sequence")
text_mod.Tokenizer = FakeTokenizer
text_mod.tokenizer_from_json = fake_tokenizer_from_json
seq_mod.pad_sequences = fake_pad_sequences
prep.text, prep.sequence = text_mod, seq_mod
keras.preprocessing = prep
tf.keras = keras
import importlib.machinery

for name, mod in [("tensorflow", tf), ("tensorflow.keras", keras),
                  ("tensorflow.keras.preprocessing", prep),
                  ("tensorflow.keras.preprocessing.text", text_mod),
                  ("tensorflow.keras.preprocessing.sequence", seq_mod)]:
    # torch._dynamo.trace_rules walks sys.modules and chokes on __spec__ = None
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    sys.modules[name] = mod

# ---------------------------------------------------------------
# Tiny local BERT instead of a downloaded bert-base-cased
# ---------------------------------------------------------------
import torch

# The stub CUDA libraries segfault if actually called; transformers probes
# torch.cuda.graphs during BERT's attention-mask construction.
torch.cuda.graphs.is_current_stream_capturing = lambda *a, **k: False
torch.cuda.is_current_stream_capturing = lambda *a, **k: False

import transformers
from transformers import BertConfig, BertModel

TINY = BertConfig(vocab_size=500, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=2, intermediate_size=64,
                  max_position_embeddings=128)


class FakeBertTok:
    def __call__(self, texts, padding=None, truncation=None, max_length=50,
                 return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = np.zeros((len(texts), max_length), dtype=np.int64)
        mask = np.zeros((len(texts), max_length), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = [abs(hash(w)) % 495 + 3 for w in str(t).split()][:max_length - 2]
            row = [101] + toks + [102]
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


os.environ["HF_HUB_OFFLINE"] = "1"
# transformers probes CUDA graph capture inside BERT's attention-mask code,
# which crashes against the stub CUDA libs.
import transformers.utils.import_utils as _tiu
_tiu.is_cuda_stream_capturing = lambda *a, **k: False
# Patch the real classes first (some transformers versions resolve the name
# through a lazy-module hook that ignores a plain setattr), then shadow them.
for _n in ("BertTokenizerFast", "BertTokenizer"):
    try:
        getattr(transformers, _n).from_pretrained = classmethod(
            lambda cls, *a, **k: FakeBertTok())
    except Exception:
        pass
transformers.BertTokenizerFast = FakeBertTok
transformers.BertTokenizer = FakeBertTok
BertModel.from_pretrained = classmethod(lambda cls, *a, **k: BertModel(TINY))

# ---------------------------------------------------------------
# Synthetic canonical dataset + splits
# ---------------------------------------------------------------
import pandas as pd

rng = np.random.default_rng(0)
VOCAB = ["good", "bad", "ok", "great", "awful", "fine", "covid", "vaccine",
         "policy", "sad", "happy", "meh", "terrible", "excellent"]
N = 420
rows = []
for i in range(N):
    lab = ["negative", "neutral", "positive"][i % 3]
    words = list(rng.choice(VOCAB, size=int(rng.integers(4, 14))))
    rows.append({"comment/tweet": " ".join(words), "majority_sent": lab})
os.makedirs("redo", exist_ok=True)
pd.DataFrame(rows).to_csv("redo/canonical_dataset.csv", index=False)

idx = rng.permutation(N)
np.savez("redo/splits.npz", train=idx[:300], val=idx[300:360], test=idx[360:])
print("synthetic data written")

# ---------------------------------------------------------------
# Run each script with shrunken budgets
# ---------------------------------------------------------------
SHRINK = {
    "N_TRIALS = 25": "N_TRIALS = 2",
    "N_TRIALS = 15": "N_TRIALS = 2",
    "MAX_EPOCHS = 25": "MAX_EPOCHS = 2",
    "MAX_EPOCHS = 8": "MAX_EPOCHS = 2",
    "TEMP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]": "TEMP_GRID = [1.0, 1.5]",
}


def run(path):
    print("\n" + "#" * 70)
    print("# RUNNING", path)
    print("#" * 70)
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    for a, b in SHRINK.items():
        src = src.replace(a, b)
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)


import matplotlib
matplotlib.use("Agg")

run("Tune GRU + Keras Word Tokenizer V3.py")
run("Tune CNN + BERT Tokenizer V3.py")
run("experiment V3 (tuned ensemble model).py")
print("\n\nALL THREE SCRIPTS COMPLETED")
