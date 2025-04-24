import base64
import json
import sys
import os

# 定义解码函数
def decode_base64(encoded_text):
    try:
        # 解码base64内容
        decoded_text = base64.b64decode(encoded_text).decode('utf-8')
        return decoded_text
    except Exception as e:
        print(f"解码失败: {e}")
        return encoded_text  # 如果解码失败，返回原始文本

# 递归处理JSON对象中的base64编码的"text"字段
def process_json(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and key == "text":
                # 仅处理"text"字段中的base64编码
                data[key] = decode_base64(value)
            else:
                # 递归处理嵌套的dict或list
                process_json(value)
    elif isinstance(data, list):
        for item in data:
            process_json(item)

# 检查命令行参数
if len(sys.argv) != 2:
    print("用法: python base64_2_utf8.py <输入文件>")
    sys.exit(1)

# 获取输入文件路径
input_file = sys.argv[1]

# 确认输入文件是否存在
if not os.path.exists(input_file):
    print(f"文件 {input_file} 不存在")
    sys.exit(1)

# 生成输出文件名 (在文件名后加 "_utf8")
file_name, file_extension = os.path.splitext(input_file)
output_file = f"{file_name}_utf8{file_extension}"

# 读取JSON文件内容
with open(input_file, 'r', encoding='utf-8') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"JSON文件解析错误: {e}")
        sys.exit(1)

# 处理JSON中的所有base64编码的"text"字段
process_json(data)

# 将解码后的JSON内容保存到新的文件中
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"解码完成，结果已保存到 {output_file}")
