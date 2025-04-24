import gradio as gr
import requests
import json
import time
from typing import Generator

# 服务器地址
SERVER_URL = "http://47.119.141.178:8000/v1/chat/completions2"

# 处理流式响应的函数
def stream_response(message: str, history: list) -> Generator[str, None, None]:
    # 设置固定的 user_id 和 session_id
    user_id = "wl_test_gdfy"
    session_id = "session_wl_session1"

    # 构造请求数据
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "content": message
    }

    # 记录开始时间
    start_time = time.time()

    # 发送 POST 请求，启用流式响应
    try:
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )
        response.raise_for_status()

        # 处理流式响应
        current_response = ""
        first_packet_time = None  # 用于记录首条报文时间

        for line in response.iter_lines():
            if line:
                # 解码并移除 "data: " 前缀
                decoded_line = line.decode("utf-8").replace("data: ", "")
                try:
                    data = json.loads(decoded_line)
                    content = data["data"]["content"]
                    think = data["data"]["think"]

                    # 更新当前响应
                    if content:
                        current_response += content
                    if think:
                        current_response += f'<span style="color:blue">{think}</span>'

                    # 如果是首条报文，记录时间
                    if first_packet_time is None:
                        first_packet_time = time.time() - start_time
                        yield f"{current_response}<br><br>首条报文响应时间: {first_packet_time:.2f}秒"
                    else:
                        # 后续报文只更新内容，不重复显示时间
                        yield f"{current_response}<br><br>首条报文响应时间: {first_packet_time:.2f}秒"
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as e:
        yield f"请求失败: {str(e)}"

# Gradio 聊天界面
def create_chat_interface():
    with gr.Blocks(title="大模型响应速度测试") as demo:
        gr.Markdown("# 大模型响应速度测试工具")
        gr.Markdown("输入消息内容，测试模型的流式响应速度。蓝色文字为 `think` 内容，仅显示首条报文时间。")

        # 使用 Chatbot 组件展示对话
        chatbot = gr.Chatbot(label="对话历史", height=400)

        # 输入框和发送按钮
        msg = gr.Textbox(label="输入消息", placeholder="请输入消息内容，例如：你好")
        submit_btn = gr.Button("发送")

        # 定义流式响应的逻辑
        def respond(message, chat_history):
            # 将用户消息添加到历史
            chat_history.append([message, None])
            # 生成流式响应
            for response in stream_response(message, chat_history):
                chat_history[-1][1] = response  # 更新最后一条消息的响应
                yield chat_history, ""  # 返回更新后的历史记录和空字符串以清空输入框

        # 点击发送按钮触发响应，并清空输入框
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],  # 输出到 chatbot 和 msg，清空输入框
        )

        # 按回车键也可以触发，并清空输入框
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],  # 输出到 chatbot 和 msg，清空输入框
        )

    return demo

# 启动 Gradio 应用，指定地址和端口
if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch(
        server_name="0.0.0.0",  # 监听地址，改为本地监听
        server_port=7676       # 监听端口
    )