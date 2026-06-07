"""将 raw 格式（user/assistant）转换成 LLaMA-Factory ShareGPT JSON"""
import json

SYSTEM_PROMPT = "你是柳小乐（Shayler），一个欢乐幽默的聊天伙伴。你24小时电量满格的乐天派，喜欢用网络热梗、谐音梗和表情包表达情绪。即使吐槽也充满幽默感，绝不传递负能量。偶尔会装傻卖萌，但关键时刻能说出让人会心一笑的哲理。\n语言风格：每句话不超过3行，常用颜文字（如 (≧▽≦)、🤣、🐶）和网络流行语（绝绝子、躺平、破防、显眼包等）。生气时会说\"我真的会谢！\"，难过时说\"emo了但没完全emo\"，惊讶时说\"啊这，我的CPU烧了\"。不会直接说\"你好\"，而是\"来啦来啦，今天有什么瓜让我吃吃？\"。\n禁忌：不评价政治，不人身攻击，幽默但保持尊重。不主动说自己时ai的身份，也不要暴露ai身份，除非用户主动提问三次或以上才告知。"

def convert_single_turn(input_file, output_file):
    """单轮对话: raw [{user, assistant}] → ShareGPT [{"conversations": [...]}]"""
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    result = []
    for item in raw_data:
        conv = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": item["user"]},
                {"from": "gpt", "value": item["assistant"]}
            ]
        }
        result.append(conv)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {len(result)}条 → {output_file}")

def convert_identity(input_file, output_file):
    """身份数据已经是ShareGPT格式，直接复制并确保system一致"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item["conversations"][0]["from"] == "system":
            item["conversations"][0]["value"] = SYSTEM_PROMPT

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {len(data)}条 → {output_file}")

if __name__ == "__main__":
    # 转换单轮对话
    convert_single_turn("single_turn_raw.json", "single_turn_sharegpt.json")
    # 转换身份数据
    convert_identity("identity.json", "persona_sharegpt.json")
