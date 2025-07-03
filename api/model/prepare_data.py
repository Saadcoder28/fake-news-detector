"""
prepare_data.py  – LIAR + FakeNewsNet
-------------------------------------
• Downloads LIAR and FakeNewsNet
• Cleans the text
• Tokenises with distilroberta-base
• Binarises labels  (0 = real, 1 = fake)
• Stratified 70 / 15 / 15 split
• Saves Arrow files to data/fake_news_splits/

Run (repo root, venv active):
    python -m api.model.prepare_data
"""

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import html, re, numpy as np, pathlib

# ── 1. Load ───────────────────────────────────────────────────────────
print("⬇️  Loading LIAR …")
liar = load_dataset("ucsbnlp/liar", split="train", trust_remote_code=True)

print("⬇️  Loading FakeNewsNet …")
fnn = load_dataset("fake-news-UFG/FakeNewsSet", split="train",
                   trust_remote_code=True)

liar = liar.rename_column("statement", "article")
text_col = "content" if "content" in fnn.column_names else "title"
fnn  = fnn.rename_column(text_col, "article")

# ── 2. Clean ──────────────────────────────────────────────────────────
def clean(t):
    t = html.unescape(t)
    t = re.sub(r"https?://\\S+|<[^>]+>", " ", t)
    t = re.sub(r"[^A-Za-z0-9 \\t]", " ", t)
    return " ".join(t.split())

liar = liar.map(lambda b: {"clean": [clean(x) for x in b["article"]]},
                batched=True, num_proc=4, load_from_cache_file=False)
fnn  = fnn .map(lambda b: {"clean": [clean(x) for x in b["article"]]},
                batched=True, num_proc=4, load_from_cache_file=False)

# ── 3. Binary-encode labels 0 / 1 ─────────────────────────────────────
named = {
    "pants-fire":1, "false":1, "barely-true":1,
    "half-true":0,  "mostly-true":0, "true":0,
    "fake":1, "real":0,
}
def to_bin(val):
    if isinstance(val, str):
        return named[val]
    # any integer >0 ➜ 1  else 0
    return 1 if int(val) else 0

liar = liar.map(lambda b: {"label_bin": [to_bin(x) for x in b["label"]]},
                batched=True, load_from_cache_file=False)
fnn  = fnn .map(lambda b: {"label_bin": [to_bin(x) for x in b["label"]]},
                batched=True, load_from_cache_file=False)

# ── 4. Tokenise ───────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained("distilroberta-base")
def tok_batch(b):
    return tok(b["clean"], truncation=True,
               padding="max_length", max_length=512)

liar = liar.map(tok_batch, batched=True,
                remove_columns=["article", "clean", "label"])
fnn  = fnn .map(tok_batch, batched=True,
                remove_columns=["article", "clean", "label"])

# ── 5. Stratified split 70/15/15 ──────────────────────────────────────
full   = concatenate_datasets([liar, fnn]).shuffle(seed=42)
y      = np.array(full["label_bin"]); idx = np.arange(len(full))
rng    = np.random.default_rng(42)
tr, va, te = [], [], []
for lb in (0,1):
    i = idx[y==lb]; rng.shuffle(i); n=len(i); a,c=int(.7*n),int(.85*n)
    tr += i[:a].tolist();  va += i[a:c].tolist();  te += i[c:].tolist()

splits = DatasetDict({
    "train": full.select(tr),
    "validation": full.select(va),
    "test": full.select(te),
})

# ── 6. Save ───────────────────────────────────────────────────────────
out = pathlib.Path("data/fake_news_splits")
out.mkdir(parents=True, exist_ok=True)
splits.save_to_disk(out.as_posix())
print("✅ Saved tokenised splits to", out)
