import json
import os
import sys
import re
import time

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
#from llama_index.llms.openai import OpenAI

from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding

from loguru import logger

#from knowledge_llama_index import LlamaIndexKnowledge
from knowledge_bank import KnowledgeBank
from keword_search import wlKeywordSearch

from qwen_agent.llm import get_chat_model
from qwen_agent.agents import Assistant

def main():
    # 记录访问大模型的初始时间
    wlstartTime = time.time()

    # 初始化 BM25 检索器
    try:
        wl_Bm25_search = wlKeywordSearch(input_dir="/root/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy")
    except Exception as e:
        logger.error(f"初始化 wlKeywordSearch 失败: {e}")
        return

    # 记录初始化耗时
    wlEndTime = time.time()
    restoreTimeCost = wlEndTime - wlstartTime
    logger.info("============wlKeywordSearch init:")
    logger.info(f"花费了实际是(s): {restoreTimeCost}", flush=True)
    logger.info("============wlKeywordSearch init:")

    # 主循环：接受用户输入并执行查询
    while True:
        query = input("请输入查询内容（输入'quit'结束）：")

        # 如果用户输入为空，提示重新输入
        if not query.strip():
            logger.info("输入不能为空，请重新输入查询内容。")
            continue

        # 如果用户输入'quit'，结束查询
        if query.lower() == 'quit':
            logger.info("退出查询")
            break

        logger.info(f"============KnowledgeBank query: {query}")

        # 记录查询开始时间
        query_start_time = time.time()

        # 调用 BM25 检索器进行搜索
        try:
            bm25_trunks_with_score = wl_Bm25_search.search(query=query, top_k=3)
            # 记录查询结束时间
            query_end_time = time.time()
            query_time_cost = query_end_time - query_start_time

            # 打印查询结果及耗时
            logger.info(f"查询完成，耗时: {query_time_cost:.4f} 秒")
            
            for chunk, score in bm25_trunks_with_score:
                logger.info(f"BM25检索结果: {chunk}，得分: {score}")
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")


if __name__ == "__main__":
    main()