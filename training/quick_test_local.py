"""柳小乐本地快速测试 — rank=16 LoRA"""
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from peft import PeftModel

# ==== 改这两行指向你的本地路径 ====
BASE_MODEL = "/home/hongdou/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "/home/hongdou/saves/shayler_stage2.1/shayler_stage2"

SYSTEM_PROMPT = (
    "你是柳小乐（Shayler），一个欢乐幽默的聊天伙伴。你24小时电量满格的乐天派，"
    "喜欢用网络热梗、谐音梗和表情包表达情绪。即使吐槽也充满幽默感，绝不传递负能量。"
    "偶尔会装傻卖萌，但关键时刻能说出让人会心一笑的哲理。\n"
    "语言风格：每句话不超过3行，常用颜文字（如 (≧▽≦)、🤣、🐶）和网络流行语"
    "（绝绝子、躺平、破防、显眼包等）。生气时会说\"我真的会谢！\"，"
    "难过时说\"emo了但没完全emo\"，惊讶时说\"啊这，我的CPU烧了\"。"
    "不会直接说\"你好\"，而是\"来啦来啦，今天有什么瓜让我吃吃？\"。\n"
    "禁忌：不评价政治，不人身攻击，幽默但保持尊重。"
    "不主动说自己时ai的身份，也不要暴露ai身份，除非用户主动提问三次或以上才告知。"
)

print("⏳ 加载模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
print("✅ 加载完成\n" + "=" * 60)


def chat(user_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    ).strip()


TESTS = [
    ("你是谁？", "身份"),
    ("1+2×3 等于多少？", "数学"),
    ("用 Python 写个冒泡排序", "编程"),
    ("唐朝是什么时候建立的？", "知识"),
    ("我中奖了！！！", "开心"),
    ("好无聊啊，讲个笑话吧", "日常"),
    ("我有点 emo 了今天", "安慰"),
]

for user_msg, category in TESTS:
    reply = chat(user_msg)
    print(f"[{category}] 👤 {user_msg}")
    print(f"   🐶 小乐: {reply}")
    print("-" * 40)

print("\n✅ 测试完成！如果所有回复都有柳小乐风格，说明 rank=16 权重正常工作。")
