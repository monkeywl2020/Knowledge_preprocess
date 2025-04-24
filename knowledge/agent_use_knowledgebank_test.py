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

  input_dir = "/root/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy"
  #input_dir = "/home/ubuntu/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy"

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
  ## 医生信息
  ## 医院信息
  ## 新生儿科信息
  ## 儿科
  ## 出生证和医保
  ## 产前诊断
  ## 放射科、超声科
  ## 门诊部
  ## 保健科、保健部
  ## 病理科
  ## 检验科
  ## 中医科
  ## 妇女保健科、妇女儿童健康
  ## 生殖科
  ## 眼科
  ## 乳腺科
  ## 营养科

# 优化后的搜索查询生成规则：
  - 1. **结合上下文**：基于用户最新输入及其上下文，判断是否涉及关注信息中的某项内容。
  - 2. **提取关键信息**：从用户输入中提炼核心关键词，避免无关内容干扰，提取的关键词放到关键词后，以[]括起来。
  - 3. **生成简洁查询**：将核心信息转化为与关注信息匹配的**简洁搜索查询**，用于快速检索相关答案。
  - 4. **无关或不明确情形**：如果用户提问与关注信息无关，或上下文不足以生成准确查询，则直接返回用户原有提问，不添加额外说明。
  - 5. **避免冗余**：生成的查询应简洁明了，不包含多余的解释或提示。
  - 6. 用户查询医院相关信息，如果没有提及具体院区，那么都是查询广东省妇幼保健医院的相关信息（明确提及清远院区，请携带院区信息）。

# 输出格式：
  - 如果相关：返回优化后的搜索查询。同时输出关键词
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
  
  # 记录访问大模型的初始时间
  wlstartTime = time.time()
  # 知识库，用于存储 LlamaIndexKnowledge 对象的容器，根据配置初始化知识库
  wlknowledge_bank = KnowledgeBank(configs = knowledge_config,
                                  emb_model = myMutilModalEmbed_model,
                                  muti_llm_model = openai_mm_llm)
  wlEndTime = time.time()
  restoreTimeCost = wlEndTime - wlstartTime

  logger.info("============KnowledgeBank restore:")
  logger.info("花费了实际是(s):",restoreTimeCost,flush=True)
  logger.info("============KnowledgeBank restore:")

  #提炼用户提问agent
  Refine_questions_agent = Assistant(llm=llm_cfg,
                  system_message=Refine_system_instruction,
                  )
# 获取知识库
wl_pdf_rag_knowledge = wlknowledge_bank.get_knowledge("wl_gdfy_rag")

def Refine_user_query(question: str, history:str = None)-> str:
    logger.info(f"LocalMultimodalRAG::Refine_user_query: ============用户查询 question:{question}")

    user_input_formatted_message = {'role': 'user', 'content': question}

    rsps = Refine_questions_agent.run(messages=[user_input_formatted_message])
    for res in rsps:
        #logger.info(f"====>res:\n {res}")
        new_qury = res
    
    new_qury_content = new_qury[0]["content"]
    logger.info(f"Refine_user_query: ============提炼后的用户查询 new_qury:{new_qury_content}")
    return new_qury_content

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

        # 计算查询字数（假设输入为中文）使用原始的 query 字符串
        #query_length = len(query.strip())
        
        # 查询开始时间
        #--------------------------------------
        wlstartTime1 = time.perf_counter()
        #--------------------------------------
        #=====================================
        # 1提炼用户查询，使用agent进行提炼
        #=====================================
        new_query = Refine_user_query(query)
        #--------------------------------------
        wlEndTime1 = time.perf_counter()
        queryTimeCost1 = wlEndTime1 - wlstartTime1
        logger.info(f"===Refine_user_query 耗时: {queryTimeCost1}秒")
        #--------------------------------------

        # 查询开始时间
        #--------------------------------------
        wlstartTime2 = time.perf_counter()
        #--------------------------------------
        #=====================================
        # BM25 检索，获取 top10 以备 RRF 使用
        #=====================================
        # 这里做BM25检索到底是用 原始用户请求做检索还是使用 提炼后的用户请求做检索 是个问题？目前gdfy用的是原始请求，效果还可以。
        bm25_trunks_with_score = wl_Bm25_search.search(query = query, top_k = 3)
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

        # Embedding 检索，获取 top10 以备 RRF 使用
        # 查询开始时间
        #--------------------------------------
        wlstartTime3 = time.time()
        #--------------------------------------
        #=====================================
        # 执行检索
        #=====================================
        nodes_with_score = wl_pdf_rag_knowledge.retrieve(query=new_query, similarity_top_k=3)  
        logger.info(f"=> nodes_with_score:{nodes_with_score}\n\n")
        
        #--------------------------------------
        # 从 NodeWithScore 中提取三元组，过滤得分低于 0.4 的结果
        SCORE_THRESHOLD = 0.35  # 定义阈值
        embedding_results: List[Tuple[str, str, float]] = [
            (node_with_score.node.metadata['file_path'], node_with_score.node.text, node_with_score.score)
            for node_with_score in nodes_with_score
            if node_with_score.score is not None and node_with_score.score >= SCORE_THRESHOLD
        ]

        # 查询结束时间
        wlEndTime3 = time.time()
        queryTimeCost3 = wlEndTime3 - wlstartTime3
        logger.info(f"==wl_pdf_rag_knowledge.retrieve=查询耗时: {queryTimeCost3}秒")
        logger.info(f"====>wl_pdf_rag_knowledge.retriev embedding_results:{embedding_results}")
        #--------------------------------------
        # 合并BM25和Embedding的结果
        combined_results = bm25_results + embedding_results

        # 去重：基于doc_path，保留得分较高的记录
        unique_results = {}
        for result in combined_results:
            doc_path, content, score = result
            if doc_path not in unique_results or score > unique_results[doc_path][2]:
                unique_results[doc_path] = (doc_path, content, score)

        # 转换为列表，保持原有顺序（不排序）
        final_combined_results = list(unique_results.values())

        # 格式化输出结果
        final_result = []
        
        logger.info("Top 检索结果（BM25 + Embedding，去重后）：")
        for idx, result in enumerate(final_combined_results, start=1):
            doc_path, content, score = result
            final_result.append(f"知识{idx}:\n- 得分: {score}\n- 内容: {content}\n")
        
        logger.info(f"最终检索结果：\n{final_result}")
        for item in final_result:
            logger.info(f"{item}")

        #logger.info(f"最终检索结果：\n{final_result}")
        if 0:
            # 找到 BM25 和 Embedding top3 的交集
            intersection = []
            for bm25_chunk in bm25_results[:3]:
                for embedding_chunk in embedding_results[:3]:
                    if is_same_chunk(bm25_chunk, embedding_chunk):
                        intersection.append(bm25_chunk)
                        break

            # 根据查询长度选择检索策略
            #if query_length <= 4:
            #    # 短输入：优先取交集，若不足3个，从 BM25 top3 补充
            #    final_results = intersection
            #    if len(final_results) < 3:
            #        remaining = 3 - len(final_results)
            #        for bm25_chunk in bm25_results[:3]:  # 只从 top3 补充
            #            if bm25_chunk not in final_results:
            #                final_results.append(bm25_chunk)
            #                remaining -= 1
            #                if remaining == 0:
            #                    break
            #else:
            # 长输入：使用 RRF 融合 BM25 和 Embedding 的 top10
            final_results = rrf_fusion(bm25_results, embedding_results)

            # 输出最终结果
            logger.info("最终检索结果：")
            for i, result in enumerate(final_results, start=1):
                doc_path, content, score = result
                logger.info(f"第{i}名：文档路径={doc_path}，得分={score}，内容={content}...")
        
        #if not new_nodes_with_score:
            #print("没有相关结果")

        #print("========================> nodes_with_score: ", new_nodes_with_score)

def main():
    # 调用查询函数
    query_knowledge()

if __name__ == "__main__":
    main()
