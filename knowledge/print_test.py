from loguru import logger

final_result = "知识1:\n- 得分: 10.156068311954401\n- 内容: #13"
print("直接打印：")
print(final_result)

logger.info(f"最终检索结果：\n{final_result}")