import json
import os
import sys
import re
import time
from typing import List, Tuple, Dict, Any

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
#from llama_index.llms.openai import OpenAI

from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding

from loguru import logger

#from knowledge_llama_index import LlamaIndexKnowledge
from knowledge_bank import KnowledgeBank
from keword_search import wlKeywordSearch

from qwen_agent.llm import get_chat_model
from qwen_agent.agents import Assistant

if 1:
  # 知识库knowledge_config 配置参说明：
  # 如果是普通pdf处理直接设置  SimpleDirectoryReader 作为 解析器。
  # 如果是需要将pdf的页面信息，还留有将pdf每一页的信息整理出来就是用 wlSimplePdfDirectoryReader， 将pdf解析成markdown格式，并且获取文档中的页面信息和截图信息
  # 两者区别是pdf处理的时候的区别。wlSimplePdfDirectoryReader是继承了SimpleDirectoryReader的，做了特殊的pdf文档处理。
  # "recursive": True, 参数表示递归遍历子目录。

  #input_dir = "/root/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy"
  input_dir = "/home/ubuntu/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy"

  #知识库bank配置
  knowledge_config =  [
    {
      "knowledge_id": "wl_gdfy_rag",
      "data_processing": [
        {
          "load_data": {
            "loader": {
              "create_object": True,
              "module": "llama_index.core",
              "class": "SimpleDirectoryReader",
              "init_args": {
                "input_dir": input_dir,
                "recursive": True,
                "required_exts": [
                  ".doc",".txt",".docx",".xlsx","xls"
                ]
              }
            }
          },
        }
      ]
    },
  ]


  Refine_system_instruction = '''你是一位专业的信息提取专家。你的任务是根据用户输入的最新信息，判断其与以下关注信息的相关性，并生成简洁、准确的搜索查询。如果输入信息与关注信息无关，则直接返回用户原有的提问，不附加任何说明。

# **关注信息**：
  ## 医生信息：
    - 2_医生信息整理_307
    - 10_新生儿科门诊医生信息

  ## 医生信息：
    - 0_医院地址信息
    - 1_科室分布_整理_235条

  ## 新生儿科：
    - 9_新生儿科护理_整理
    - 11_新生儿医生问答_整理

  ## 儿科：
    - 6_儿科常见问题_整理
    - 7_儿科护理_问题整理
    - 16_儿童内分泌遗传代谢科_问题整理
    - 17_胎儿医学全生命周期门诊问题
  
  ## 出生证和医保：
    - 5_出生证常见问题_整理 
    - 18_医保科_问题整理

  ## 产前诊断：
    - 13_产前诊断护理-问题_整理
    - 14_产前诊断门诊问题_整理
    - 22_产前诊断_基层医院问题征集10

  ## 放射科：
    - 19_放射科问答_整理
    - 20_放射科护理_整理
  
  ## 门诊部：
    - 00_门诊部线上组客服_问题整理_136
  
  ## 保健部：
    - 3_保健部_问题整理

  ## 病理科：
    - 4_病理科_问题整理

  ## 检验科：
    - 8_检验科问题_整理

  ## 越秀急诊科：
    - 12_越秀急诊科_问题整理

  ## 清远院区：
    - 21_清远院区_专属问题_整理
      
# 优化后的搜索查询生成规则：
  - 1. **结合上下文**：基于用户最新输入及其上下文，判断是否涉及关注信息中的某项内容。
  - 2. **提取关键信息**：从用户输入中提炼核心关键词，避免无关内容干扰。
  - 3. **生成简洁查询**：将核心信息转化为与关注信息匹配的**简洁搜索查询**，用于快速检索相关答案。
  - 4. **无关或不明确情形**：如果用户提问与关注信息无关，或上下文不足以生成准确查询，则直接返回用户原有提问，不添加额外说明。
  - 5. **避免冗余**：生成的查询应简洁明了，不包含多余的解释或提示。

# 输出格式：
  - 如果相关：返回优化后的搜索查询。
  - 如果无关或不明确：直接返回用户原始提问。'''

  llm_cfg = {
      'model': '/data/cyy/models/Qwen2___5-72B-Instruct-AWQ',
      'model_server': 'http://39.105.152.35:8980/v1',  
      'api_key': 'EMPTY',

      # (Optional) LLM hyperparameters for generation:
      'generate_cfg': {
          'top_p': 0.8,
          'temperature':0.1,
      }
  }
  #--------------------------------------------------------------
  # 多模态 Embedding
  myMutilModalEmbed_model = wlMultiModalEmbedding(
      model_name = '/models/bge-m3',
      model_weight = '/models/bge-visualized/Visualized_m3.pth'
  )

  # 多模态 llm,用来回答多模态问题的，当前pdf没使用
  local_chatLLm_cfg = {
      "model":"/data2/models/Qwen2.5-72B-Instruct-AWQ",  # 使用您的模型名称
      "api_base":"http://172.21.30.230:8980/v1/",  # 您的 vllm 服务器地址
      "api_key":"EMPTY",  # 如果需要的话
  }
  # define our lmms ,采用本地 qwen2-vl大模型
  openai_mm_llm = OpenAIMultiModal(**local_chatLLm_cfg)
#--------------------------------------------------------------
#  首先，创建知识库，这个会使用llama-index对文档进行解析，形成索引，
#  首次加载会比较慢，后续会使用这个已经建立的索引文件恢复索引
#--------------------------------------------------------------#

  # 记录访问大模型的初始时间
  wlstartTime = time.time()
  # 新增一个 BM25 检索器，使用关键字搜索
  wl_Bm25_search = wlKeywordSearch(input_dir = input_dir)
  #wl_Bm25_search = wlKeywordSearch(input_dir = "/home/ubuntu/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy")

  wlEndTime = time.time()
  restoreTimeCost = wlEndTime - wlstartTime
  logger.info("============wlKeywordSearch init:")
  logger.info("花费了实际是(s):",restoreTimeCost,flush=True)
  logger.info("============wlKeywordSearch init:")
  

# 定义函数：判断两个分块是否相同
def is_same_chunk(bm25_chunk, embedding_chunk):
    bm25_path, bm25_content, _ = bm25_chunk
    embedding_path, embedding_content, _ = embedding_chunk
    #return bm25_path == embedding_path and bm25_content.strip() == embedding_content.strip()
    # 只比较路径是否相同，不比较内容影响效率，人为保证文档路径唯一性
    return bm25_path == embedding_path 

# 长输入：使用 RRF 融合 BM25 和 Embedding 的 top10
def rrf_fusion(bm25_top: List[Tuple[str, str, float]], 
               embedding_top: List[Tuple[str, str, float]], 
               k: int = 60,
               w_bm25: float = 1,
               w_embedding: float = 1
               ) -> List[Tuple[str, str, float]]:
    rrf_scores: Dict[str, float] = {}

    # BM25 排名得分
    for rank, chunk in enumerate(bm25_top, start=1):
        path = chunk[0]  # 只使用路径作为键
        if path not in rrf_scores:
            rrf_scores[path] = 0
        rrf_scores[path] += w_bm25 * 1 / (k + rank)

    # Embedding 排名得分
    for rank, chunk in enumerate(embedding_top, start=1):
        path = chunk[0]  # 只使用路径作为键
        if path not in rrf_scores:
            rrf_scores[path] = 0
        rrf_scores[path] += w_embedding * 1 / (k + rank)

    # 按 RRF 分数排序
    sorted_paths = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 恢复完整三元组
    final_chunks = []
    for path, _ in sorted_paths:
        # 优先从 Embedding 获取完整信息，若无则从 BM25，先从Embedding中找，找不到在next到 bm25中找
        chunk = next((c for c in embedding_top if c[0] == path),  # 使用生成器表达式遍历 embedding_top（Embedding 召回的结果列表）
                    next((c for c in bm25_top if c[0] == path), None)) #在 bm25_top（BM25 召回的结果列表）中查找与 path 匹配的文档块。
        if chunk:
            final_chunks.append(chunk)

    # 强制保留BM25的top1
    bm25_top1 = bm25_top[0]  # BM25的第1名
    if bm25_top1 not in final_chunks[:3]:
        final_chunks = [bm25_top1] + final_chunks[:2]  # 插入top1，保留前2个

    logger.info(f"RRF final_chunks: {final_chunks}")
    return final_chunks[:3]  # 返回top3

def query_knowledge():

    # 循环等待用户输入查询
    while True:
        query = input("请输入查询内容（输入'quit'结束）：")

        # 如果用户输入为空，提示重新输入
        if not query.strip():
            logger.info("输入不能为空，请重新输入查询内容。")
            continue

        # 如果用户输入'退出'，结束查询
        if query.lower() == 'quit':
            logger.info("退出查询")
            break

        logger.info(f"============KnowledgeBank query: {query}")

        # 查询开始时间
        #--------------------------------------
        wlstartTime2 = time.perf_counter()
        #--------------------------------------
        #=====================================
        # BM25 检索，获取 top10 以备 RRF 使用
        #=====================================
        bm25_trunks_with_score = wl_Bm25_search.search(query = query, top_k = 10)
        # 提取BM25检索结果的三元组（路径，内容，得分）列表
        bm25_results = [
            (chunk.metadata['source'], chunk.content, score)
            for chunk, score in bm25_trunks_with_score
        ]
        #--------------------------------------
        wlEndTime2 = time.perf_counter()
        queryTimeCost2 = wlEndTime2 - wlstartTime2
        logger.info(f"===wl_Bm25_search.search 耗时: {queryTimeCost2}秒")
        logger.info(f"===wl_Bm25_search.search results: {bm25_results}")
        #--------------------------------------

        # ====== 新增：倒排打印 BM25 结果 ======
        logger.info("====== 倒排打印 BM25 检索结果 ======")
        # 对结果逆序处理（保持原列表不变，生成新列表）
        reversed_results = bm25_results[::-1]  # 或使用 reversed(bm25_results)
        for idx, (source, content, score) in enumerate(reversed_results, 1):
            logger.info(f"结果 {idx}:")
            logger.info(f"来源: {source}")
            logger.info(f"内容: \n{content}")
            logger.info(f"得分: {score}")
            logger.info("-" * 40)  # 分隔线
        # ====================================

def main():
    # 调用查询函数
    query_knowledge()

if __name__ == "__main__":
    main()
