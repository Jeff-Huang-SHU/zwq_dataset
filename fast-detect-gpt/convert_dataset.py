import json
import argparse
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer


def normalize_text(text: str) -> str:
    """
    简单清洗文本：
    1. 去掉首尾空白
    2. 把多行文本压成单行
    """
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def truncate_by_tokens(text: str, tokenizer, max_tokens: int):
    """
    按 tokenizer 的 token 数截断文本。

    返回：
    truncated_text: 截断后的文本
    original_len: 原始 token 数
    truncated: 是否发生截断
    """
    input_ids = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False
    )["input_ids"]

    original_len = len(input_ids)

    if original_len <= max_tokens:
        return text, original_len, False

    truncated_ids = input_ids[:max_tokens]

    truncated_text = tokenizer.decode(
        truncated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return truncated_text, original_len, True


def convert_format_a_to_b(
    input_path,
    output_path,
    tokenizer_name="EleutherAI/gpt-neo-2.7B",
    cache_dir="./cache",
    max_tokens=1024,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir
    )

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("输入 JSON 必须是一个 list，例如：[ {...}, {...}, ... ]")

    # 按 register + tittle 分组
    groups = defaultdict(lambda: {"human": [], "ai": []})

    total_records = 0
    skipped_empty_content = 0

    for rec in records:
        total_records += 1

        register = rec.get("register", "")
        tittle = rec.get("tittle", "")  # 注意：你的字段名是 tittle，不是 title
        content = normalize_text(rec.get("content", ""))
        model = rec.get("model", "")

        if not content:
            skipped_empty_content += 1
            continue

        key = (register, tittle)

        if model == "h":
            groups[key]["human"].append(content)
        else:
            groups[key]["ai"].append(content)

    original = []
    sampled = []

    group_stats = []

    total_pairs = 0
    truncated_human_count = 0
    truncated_ai_count = 0

    for key, pair_data in groups.items():
        human_texts = pair_data["human"]
        ai_texts = pair_data["ai"]

        num_pairs = min(len(human_texts), len(ai_texts))

        if num_pairs == 0:
            group_stats.append({
                "register": key[0],
                "tittle": key[1],
                "human_count": len(human_texts),
                "ai_count": len(ai_texts),
                "paired_count": 0,
                "status": "skipped"
            })
            continue

        group_truncated_human = 0
        group_truncated_ai = 0

        for i in range(num_pairs):
            h_text = human_texts[i]
            a_text = ai_texts[i]

            h_text, h_len, h_truncated = truncate_by_tokens(
                h_text,
                tokenizer,
                max_tokens
            )
            a_text, a_len, a_truncated = truncate_by_tokens(
                a_text,
                tokenizer,
                max_tokens
            )

            if h_truncated:
                truncated_human_count += 1
                group_truncated_human += 1

            if a_truncated:
                truncated_ai_count += 1
                group_truncated_ai += 1

            original.append(h_text)
            sampled.append(a_text)
            total_pairs += 1

        group_stats.append({
            "register": key[0],
            "tittle": key[1],
            "human_count": len(human_texts),
            "ai_count": len(ai_texts),
            "paired_count": num_pairs,
            "truncated_human_count": group_truncated_human,
            "truncated_ai_count": group_truncated_ai,
            "status": "paired"
        })

    output_data = {
        "original": original,
        "sampled": sampled
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 额外保存一个配对统计文件，方便你检查哪些组没配上
    stats_path = output_path.with_suffix(".pair_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(group_stats, f, ensure_ascii=False, indent=2)

    print("转换完成")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"配对统计: {stats_path}")
    print()
    print(f"总记录数: {total_records}")
    print(f"空 content 跳过数: {skipped_empty_content}")
    print(f"最终 paired 样本数: {len(original)}")
    print(f"original 数量: {len(original)}")
    print(f"sampled 数量: {len(sampled)}")
    print(f"max_tokens: {max_tokens}")
    print(f"被截断的人类文本数: {truncated_human_count}")
    print(f"被截断的 AI 文本数: {truncated_ai_count}")

    if len(original) != len(sampled):
        raise RuntimeError("错误：original 和 sampled 数量不一致")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入格式A JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出格式B JSON文件路径"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="EleutherAI/gpt-neo-2.7B",
        help="用于计算 token 长度和截断的 tokenizer"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Hugging Face 模型缓存目录"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="每条文本最多保留多少个 token"
    )

    args = parser.parse_args()

    convert_format_a_to_b(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()