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

def read_questions(file_path):
    """读取问题文件中的问题内容"""
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 假设每行格式为 "编号. 问题内容"，提取问题内容
                match = re.match(r'^\d+\.\s*(.+)$', line.strip())
                if match:
                    question = match.group(1)
                    questions.append(question)
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
    return questions

def check_question_in_results(question, final_result):
    """检查问题是否出现在 final_result 的内容中"""
    for item in final_result:
        if question in item:
            return True
    return False

def process_folder(input_folder, output_folder, wl_Bm25_search, wl_pdf_rag_knowledge, Refine_user_query):
    """遍历输入文件夹及其子文件夹，处理所有txt文件，检查问题并保存未找到的问题"""
    if not os.path.isdir(input_folder):
        logger.error(f"{input_folder} 不是一个有效的文件夹路径")
        return

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹及其子文件夹
    for root, _, files in os.walk(input_folder):
        # 计算相对路径并创建对应的输出子文件夹
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for file_name in files:
            if file_name.lower().endswith('.txt'):
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(output_subfolder, f"unfound_{file_name}")
                logger.info(f"处理文件: {input_file_path}")
                
                # 读取问题
                questions = read_questions(input_file_path)
                if not questions:
                    logger.warning(f"在 {input_file_path} 中未找到任何问题")
                    continue

                # 保存未找到的问题
                unfound_questions = []
                for query in questions:
                    logger.info(f"============KnowledgeBank query: {query}")

                    # 1. 提炼用户查询
                    wlstartTime1 = time.perf_counter()
                    new_query = Refine_user_query(query)
                    wlEndTime1 = time.perf_counter()
                    queryTimeCost1 = wlEndTime1 - wlstartTime1
                    logger.info(f"===Refine_user_query 耗时: {queryTimeCost1}秒")

                    # 2. BM25 检索
                    wlstartTime2 = time.perf_counter()
                    bm25_trunks_with_score = wl_Bm25_search.search(query=query, top_k=3)
                    bm25_results = [
                        (chunk.metadata['source'], chunk.content, score)
                        for chunk, score in bm25_trunks_with_score
                    ]
                    wlEndTime2 = time.perf_counter()
                    queryTimeCost2 = wlEndTime2 - wlstartTime2
                    logger.info(f"===wl_Bm25_search.search 耗时: {queryTimeCost2}秒")
                    logger.info(f"===wl_Bm25_search.search results: {bm25_results}")

                    # 3. Embedding 检索
                    wlstartTime3 = time.time()
                    nodes_with_score = wl_pdf_rag_knowledge.retrieve(query=new_query, similarity_top_k=3)
                    SCORE_THRESHOLD = 0.35
                    embedding_results = [
                        (node_with_score.node.metadata['file_path'], node_with_score.node.text, node_with_score.score)
                        for node_with_score in nodes_with_score
                        if node_with_score.score is not None and node_with_score.score >= SCORE_THRESHOLD
                    ]
                    wlEndTime3 = time.time()
                    queryTimeCost3 = wlEndTime3 - wlstartTime3
                    logger.info(f"==wl_pdf_rag_knowledge.retrieve 耗时: {queryTimeCost3}秒")
                    logger.info(f"====>wl_pdf_rag_knowledge.retrieve embedding_results: {embedding_results}")

                    # 4. 合并 BM25 和 Embedding 结果
                    combined_results = bm25_results + embedding_results
                    unique_results = {}
                    for result in combined_results:
                        doc_path, content, score = result
                        if doc_path not in unique_results or score > unique_results[doc_path][2]:
                            unique_results[doc_path] = (doc_path, content, score)
                    final_combined_results = list(unique_results.values())

                    # 5. 格式化输出结果
                    final_result = []
                    logger.info("Top 检索结果（BM25 + Embedding，去重后）：")
                    for idx, result in enumerate(final_combined_results, start=1):
                        doc_path, content, score = result
                        final_result.append(f"知识{idx}:\n- 得分: {score}\n- 内容: {content}\n")
                    logger.info(f"最终检索结果：\n{final_result}")

                    # 6. 检查问题是否在结果中
                    if check_question_in_results(query, final_result):
                        logger.info(f"问题 '{query}' 在检索结果中找到")
                    else:
                        logger.warning(f"问题 '{query}' 未在检索结果中找到")
                        unfound_questions.append(query)

                # 仅当有未找到的问题时才生成输出文件
                if unfound_questions:
                    try:
                        with open(output_file_path, 'w', encoding='utf-8') as output_file:
                            for idx, question in enumerate(unfound_questions, 1):
                                output_line = f"{idx}. {question}\n"
                                output_file.write(output_line)
                        logger.info(f"未找到的问题已写入: {output_file_path}")
                    except Exception as e:
                        logger.error(f"写入文件 {output_file_path} 时出错: {e}")
                else:
                    logger.info(f"所有问题在 {input_file_path} 中均找到，无未找到的问题")

def main():
    # 检查命令行参数
    # 用法: 这个文件是用来批量测试知识库的，使用Bm25算法+Embedding算法，根据输入的问题的文件夹 里面全部是问题。输出哪些检索不到的问题放到输出文件夹中去。
    # 示例: python agent_knowledgebank_batch_test.py /path/to/input_folder /path/to/output_folder
    if len(sys.argv) != 3:
        logger.error("用法: python script.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"输出文件夹: {output_folder}")

    process_folder(input_folder, output_folder, wl_Bm25_search, wl_pdf_rag_knowledge, Refine_user_query)
    logger.info("处理完成")

if __name__ == "__main__":
    main()
