import json
from transformers import AutoTokenizer

path = "《《待检查数据集路径》》"
# 例如：
# path = "exp_zwq/data/zwq_uer-gpt2-chinese.raw_data.json"
tok = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir="./cache")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

max_item = ("", -1, 0)
bad = []

for split in ["original", "sampled"]:
    for i, text in enumerate(data[split]):
        n = len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        if n > max_item[2]:
            max_item = (split, i, n)
        if n > 1024:
            bad.append((split, i, n))

print("original 数量:", len(data["original"]))
print("sampled 数量:", len(data["sampled"]))
print("最长文本:", max_item)
print("超过 1024 的数量:", len(bad))