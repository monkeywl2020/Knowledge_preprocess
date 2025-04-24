import os
import sys
import re
from loguru import logger

  # 初始化 wl_Bm25_search（需要从你的代码中引入）
from keword_search import wlKeywordSearch

# 知识库目录
input_dir = "/home/ubuntu/wlwork/wl_Knowledge_preprocess/knowledge/knowledge_gdfy"  # 根据你的配置

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

def check_question_in_results(question, bm25_results):
    """检查问题是否出现在 BM25 检索结果的内容中"""
    for _, content, _ in bm25_results:
        if question in content:
            return True
    return False

def process_folder(input_folder, output_folder, wl_Bm25_search):
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
                for question in questions:
                    logger.info(f"查询问题: {question}")
                    # 使用 BM25 检索
                    bm25_trunks_with_score = wl_Bm25_search.search(query=question, top_k=3)
                    bm25_results = [
                        (chunk.metadata['source'], chunk.content, score)
                        for chunk, score in bm25_trunks_with_score
                    ]
                    
                    # 检查问题是否在结果中
                    if check_question_in_results(question, bm25_results):
                        logger.info(f"问题 '{question}' 在检索结果中找到")
                    else:
                        logger.warning(f"问题 '{question}' 未在检索结果中找到")
                        unfound_questions.append(question)

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
    # 用法: 这个文件是用来批量测试知识库的，使用Bm25算法,根据输入的问题的文件夹 里面全部是问题。输出哪些检索不到的问题放到输出文件夹中去。
    # 示例: python bm25_knowledgebank_batch_test.py /path/to/input_folder /path/to/output_folder
    if len(sys.argv) != 3:
        logger.error("用法: python script.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"输出文件夹: {output_folder}")
    
    wl_Bm25_search = wlKeywordSearch(input_dir=input_dir)
    logger.info("wlKeywordSearch 初始化完成")

    process_folder(input_folder, output_folder, wl_Bm25_search)
    logger.info("处理完成")

if __name__ == "__main__":
    main()