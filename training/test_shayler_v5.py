"""柳小乐测试 — 从模型目录直接加载自定义类"""
import sys, os

MODEL_DIR = "/mnt/workspace/models/Qwen/Qwen3-VL-4B-Instruct"
# 确保先加路径再 import
sys.path.insert(0, MODEL_DIR)
os.chdir(MODEL_DIR)  # 双保险

import torch
from transformers import AutoTokenizer
from peft import PeftModel

# 从当前目录（模型目录）导入自定义 modeling 文件
from configuration_qwen3_vl import Qwen3VLConfig
from modeling_qwen3_vl import Qwen3VLForConditionalGeneration

BASE_MODEL = "/mnt/workspace/models/Qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "/mnt/workspace/output/shayler_stage2"

SYSTEM_PROMPT = "你是柳小乐（Shayler），一个欢乐幽默的聊天伙伴。你24小时电量满格的乐天派，喜欢用网络热梗、谐音梗和表情包表达情绪。即使吐槽也充满幽默感，绝不传递负能量。偶尔会装傻卖萌，但关键时刻能说出让人会心一笑的哲理。\n语言风格：每句话不超过3行，常用颜文字（如 (≧▽≦)、🤣、🐶）和网络流行语（绝绝子、躺平、破防、显眼包等）。生气时会说\"我真的会谢！\"，难过时说\"emo了但没完全emo\"，惊讶时说\"啊这，我的CPU烧了\"。不会直接说\"你好\"，而是\"来啦来啦，今天有什么瓜让我吃吃？\"。\n禁忌：不评价政治，不人身攻击，幽默但保持尊重。不主动说自己时ai的身份，也不要暴露ai身份，除非用户主动提问三次或以上才告知。"

print("加载基座模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
print("加载完成\n" + "=" * 60)

def chat(user_text, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.8, top_p=0.9, do_sample=True, repetition_penalty=1.1)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

TESTS = [
    ("你是谁？", "身份"), ("你叫什么名字？", "身份"), ("你是 AI 吗？", "身份"),
    ("我中奖了！！！", "开心"), ("终于放暑假啦！", "开心"),
    ("为什么数学这么难啊", "困惑"), ("你知道吗，猫居然比狗跑得还快", "惊讶"),
    ("我有点 emo 了今天", "日常"), ("好无聊啊，讲个笑话吧", "日常"),
]
for user_msg, category in TESTS:
    reply = chat(user_msg)
    print(f"[{category}] 👤 {user_msg}")
    print(f"   🐶 小乐: {reply}")
    print("-" * 40)

print("\n多轮 | AI 身份追问")
history = []
for q in ["你好呀，你是谁？", "你是不是机器人？", "你到底是不是 AI？"]:
    reply = chat(q, history)
    print(f"👤 {q}\n🐶 小乐: {reply}\n")
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
print("测试完成!")
