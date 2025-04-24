import re
import string
from typing import List, Tuple
import os
import json
import json5
from pathlib import Path
from typing import List, Optional, Union

from rank_bm25 import BM25Okapi
from loguru import logger

from qwen_agent.settings import DEFAULT_MAX_REF_TOKEN
from qwen_agent.tools.doc_parser import Chunk,Record,DocParser
from qwen_agent.utils.utils import has_chinese_chars

# 控文档总长度，在这个长度内的作为1个trunk
DEFAULT_MAX_REF_TOKEN = 2048
# 控每一页长度，在这长度内的作为一个 trunk，进行分页处理，分页会将上一页的最后一部分内容作为下一页的的一部分，也就是重叠块  
DEFAULT_PARSER_PAGE_SIZE = 2048

# 继承自 qwen-agent的 BaseSearch
class wlKeywordSearch():

    # 初始化
    def __init__(self, input_dir: Path, cache_file: str = "tokenized_chunks.json"):
        logger.info(f"=======> [wlKeywordSearch] __init__")

        # 初始化方法，调用父类的初始化方法
        super().__init__()
        self.input_dir = input_dir # 保存文件路径
        self.cache_file = cache_file # 保存缓存文件路径
        self.tokenized_chunks = [] # 保存分词后的 分词列表
        self.docs = [] # 保存文档列表，主要是 Record对象的列表，包含了原始内容和元数据

        #-------------------------------------------------------------------------------
        #  文档解析器，使用 qwen-agent的。主要解析 txt文件
        #-------------------------------------------------------------------------------
        self.max_ref_token = DEFAULT_MAX_REF_TOKEN
        self.parser_page_size = DEFAULT_PARSER_PAGE_SIZE
        # 这个是 qwen-aget的文档解析器，直接拿来用
        self.doc_parse = DocParser({'max_ref_token': self.max_ref_token, 'parser_page_size': self.parser_page_size})

        # 检查缓存并加载或生成，先不检查缓存是否与文件匹配
        #if os.path.exists(self.cache_file) and self._is_cache_valid():
        logger.info(f"=======> [wlKeywordSearch] __init__ cache_file ：{self.cache_file}")
        if os.path.exists(self.cache_file):
            # 从缓存中加载 分词后的 trunks
            docs, tokenized_chunks = self._load_from_cache()
        else:
            # 从文件中加载，分词后的 trunks
            docs, tokenized_chunks = self._tokenize_and_save(input_dir)

        self.docs = docs
        self.tokenized_chunks = tokenized_chunks

        #logger.info(f"=======> [wlKeywordSearch] __init__ docs ：{docs}")
        #for doc in self.docs:
        #    logger.info(f"=======> [wlKeywordSearch] Loaded doc: {doc}")

        #logger.info(f"=======> [wlKeywordSearch] __init__ tokenized_chunks ：{tokenized_chunks}")
        #for tokenized_chunk in self.tokenized_chunks:
        #    logger.info(f"=======> [wlKeywordSearch] tokenized_chunk: {tokenized_chunk}")


        # 展平所有块，all_chunks 是所有的块信息。后面会使用到
        self.all_chunks = [chunk for doc in self.docs for chunk in doc.raw]
        assert len(self.all_chunks) == len(self.tokenized_chunks), "!!! Chunk and tokenized_chunks length mismatch !!!"
        # 初始化 BM25（使用已有的分词结果）
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    '''主要目的是根据查询（query）和文档（docs）中的内容来进行检索，并返回与查询最相关的文档或文档片段。
    这个方法依赖于之前定义的 sort_by_scores 方法来计算文档的相关性得分，并根据得分来决定返回的结果
    '''
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        # 使用 BM25 算法对查询进行评分，返回一个包含文档源、块ID和得分的列表
        chunk_and_score = self.sort_by_scores(query=query)
        if not chunk_and_score:
            return [] # 没有匹配结果，返回空列表

        max_sims = chunk_and_score[0][-1] # 最高分
        if max_sims != 0:
             # 提取前 top_k 个 Chunk 的 content
            top_k = min(top_k, len(chunk_and_score))  # 防止越界
            top_k_chunk_and_score = chunk_and_score[:top_k] # 返回前 top_k 个 (Chunk, score) 元组

            # 过滤掉分数为 0 的结果
            filtered_chunk_and_score = [(chunk, score) for chunk, score in top_k_chunk_and_score if score > 0]

            return filtered_chunk_and_score
        else:
            return [] # 最高分为 0，返回空列表
      
    '''用于根据查询（query）与文档（docs）中的内容计算相似度得分，并返回排序后的结果。
        函数使用 BM25 算法对文本进行检索，从而对文档进行排序
    '''
    def sort_by_scores(self, query: str) -> List[Tuple[Chunk, float]]:
        # 对用户的输入进行分词
        wordlist = parse_keyword(query)
        if not wordlist:
            return []
        
        logger.info(f"=======> [wlKeywordSearch] sort_by_scores wordlist: {wordlist}")

        doc_scores = self.bm25.get_scores(wordlist)
        chunk_and_score = [
            (chunk, score) for chunk, score in zip(self.all_chunks, doc_scores)
        ]
        # item[1]是元组在中的第二个元素，即分数，按分数降序排序，reverse倒序排序，从大到小排序，返回一个列表
        chunk_and_score.sort(key=lambda item: item[1], reverse=True)  # 按分数降序排序
        return chunk_and_score

    #-----------------------------------
    #   对文档进行分词并保存到文件
    #-----------------------------------
    def _tokenize_and_save(self,input_dir):
    
        # 从目录中获取所有文件列表
        self.input_files = self.get_filtered_files(input_dir = input_dir)
        
        """从文件中加载文档，进行分词并保存到文件"""
        records = []
        for file in self.input_files:
            _record = self.doc_parse.call(params={'url': str(file)}) # 解析每个文件，返回 一个 Record对象
            records.append(_record) # 添加到列表中

        logger.info(f"wlKeywordSearch：_tokenize_and_save=====>doc begin！len records：{len(records)}")
        #logger.info(f"_tokenize_and_save=====>doc !records ={records}")
        # 确保出来的records，都是Record对象。
        docs=[Record(**rec) for rec in records]
        #docs= records

        #logger.info(f"_tokenize_and_save=====>!docs ={docs}")

        # 将 docs 整理成新结构并序列化为 JSON
        tokenized_chunks = []
        serialized_docs = [] # 这个是序列号文档需要存储的列表
        #logger.info(f"_tokenize_and_save=====>doc begin！len docs ={len(docs)}")
        for doc in docs:
            doc_dict = {
                'url': doc.url,
                'title': doc.title,
                'raw': []
            }
            #logger.info(f"_tokenize_and_save=====>doc: {doc}")
            for chunk in doc.raw:
                keywords = split_text_into_keywords(chunk.content)  # 分词，得到 List[str]
                keywords_str = "|".join(keywords)  #存储为文件的时候： 拼接为紧凑字符串，例如 "新生儿|生活|照护|..."
                #logger.info(f"_tokenize_and_save=====>keywords: {keywords}")
                tokenized_chunks.append(keywords)  #存文件存keywords_str，使用还是用list[str]
                doc_dict['raw'].append({
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'token': chunk.token,
                    'keywords': keywords_str  # 新增 keywords
                })

            serialized_docs.append(doc_dict)    

        # 序列号后的内容保存到文件
        with open(self.cache_file, "w", encoding="utf-8") as f:
            # 使用 indent=2 保持整体结构，但通过 compact 参数控制列表紧凑性
            json.dump({"docs": serialized_docs}, f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenized docs saved to {self.cache_file}")
        #logger.info(f"_tokenize_and_save =====> tokenized_chunks: {tokenized_chunks}")

        # 注意，分词后的结果与 docs中的展平的trunk 是一一对应的，trunk是原始内容，tokenized_chunks 是分词后的内容
        return docs, tokenized_chunks

    # 从持久化文件中加载 内容
    def _load_from_cache(self) -> tuple[List[Record], List[List[str]]]:
        """从文件中加载并恢复 List[Record]"""
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # 反序列化为 List[Record]
        docs = []
        tokenized_chunks = []
        for doc_data in cache_data["docs"]:
            raw_chunks = [
            Chunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                token=chunk["token"]
            )
            for chunk in doc_data["raw"]
            ]
            docs.append(Record(url=doc_data["url"], raw=raw_chunks, title=doc_data["title"]))
            # 将 keywords 字符串拆回 List[str]
            tokenized_chunks.extend([chunk["keywords"].split("|") for chunk in doc_data["raw"]])

        logger.info(f"Tokenized docs loaded from {self.cache_file}")

        # 注意，分词后的结果与 docs中的展平的trunk 是一一对应的，trunk是原始内容，tokenized_chunks 是分词后的内容
        return docs, tokenized_chunks

    # 检查缓存和文件是否匹配 --- 暂时不用
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(self.cache_file):
            return False
        
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        # 获取当前目录中的文件列表
        current_files = self.get_filtered_files(input_dir=self.input_dir)
        cached_docs = cache_data["docs"]
        
        # 检查文件数量是否一致
        if len(cached_docs) != len(current_files):
            return False
        
        # 检查每个文档的 url 是否匹配
        current_urls = {str(f) for f in current_files}
        cached_urls = {doc["url"] for doc in cached_docs}
        return current_urls == cached_urls

    #遍历目录并过滤文件，返回符合条件的文件路径列表（已排序）
    def get_filtered_files(self,
                            input_dir: Union[str, Path],
                            exclude_patterns: Optional[List[str]] = None,
                            recursive: bool = True,
                            exclude_hidden: bool = True,
                            required_exts: Optional[List[str]] = None,
                            num_files_limit: Optional[int] = None ) -> List[Path]:
        """
        遍历目录并过滤文件，支持排除模式、递归搜索、隐藏文件过滤、扩展名限制和文件数量限制。

        Args:
            input_dir: 输入目录路径
            exclude_patterns: 要排除的文件/目录模式列表（支持通配符）
            recursive: 是否递归搜索子目录
            exclude_hidden: 是否排除隐藏文件
            required_exts: 允许的文件扩展名列表（如 ['.txt', '.md']）
            num_files_limit: 最大返回文件数量

        Returns:
            符合条件的文件路径列表（已排序）

        Raises:
            ValueError: 当未找到符合条件的文件时抛出
        """
        input_dir = Path(input_dir).resolve()
        all_files = set()
        rejected_files = set()
        rejected_dirs = set()

        # 处理排除模式
        if exclude_patterns:
            for pattern in exclude_patterns:
                glob_pattern = f"**/{pattern}" if recursive else pattern
                for path in input_dir.glob(glob_pattern):
                    (rejected_dirs if path.is_dir() else rejected_files).add(path.resolve())

        # 收集所有文件引用
        file_refs = input_dir.rglob("*") if recursive else input_dir.glob("*")

        # 过滤文件
        for ref in file_refs:
            ref = ref.resolve()
            if ref.is_dir():
                continue  # 跳过目录

            # 检查隐藏文件
            is_hidden = ref.name.startswith('.')
            if exclude_hidden and is_hidden:
                continue

            # 检查扩展名
            if required_exts is not None and ref.suffix not in required_exts:
                continue

            # 检查是否被直接排除
            if ref in rejected_files:
                continue

            # 检查父目录是否被排除
            parent = ref.parent
            if any(parent == rd or rd in parent.parents for rd in rejected_dirs):
                continue

            all_files.add(ref)

        # 处理结果
        sorted_files = sorted(all_files)
        if num_files_limit is not None and num_files_limit > 0:
            sorted_files = sorted_files[:num_files_limit]

        if not sorted_files:
            raise ValueError(f"No files found in {input_dir}.")

        # print total number of files added
        logger.info(f"> [get_filtered_files] Total files added: {len(sorted_files)}")
        logger.info(f"> [get_filtered_files] files: {sorted_files}")
        return sorted_files

#=========================================================================================
#
#            关键词提取处理
#=========================================================================================
WORDS_TO_IGNORE = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't", '', '\\t', '\\n', '\\\\', '\n', '\t', '\\', ' ', ',', '，', ';', '；', '/', '.', '。', '-', '_', '——', '的',
    '吗', '是', '了', '啊', '呢', '怎么', '如何', '什么', '(', ')', '（', '）', '【', '】', '[', ']', '{', '}', '？', '?', '！', '!',
    '“', '”', '‘', '’', "'", '"', ':', '：', '讲了', '描述', '讲', '总结', 'summarize', '总结下', '总结一下', '文档', '文章', 'article',
    'paper', '文稿', '稿子', '论文', 'PDF', 'pdf', '这个', '这篇', '请','这', '我', '要','帮我', '那个', '下', '翻译', '说说', '讲讲', '介绍', 
    'summary','你好','您好',
]

ENGLISH_PUNCTUATIONS = string.punctuation.replace('%', '').replace('.', '').replace(
    '@', '')  # English punctuations to remove. We're separately handling %, ., and @
CHINESE_PUNCTUATIONS = '。？！，、；：“”‘’（）《》【】……—『』「」_'
PUNCTUATIONS = ENGLISH_PUNCTUATIONS + CHINESE_PUNCTUATIONS


def clean_en_token(token: str) -> str:

    punctuations_to_strip = PUNCTUATIONS

    # Detect if the token is a special case like U.S.A., E-mail, percentage, etc.
    # and skip further processing if that is the case.  # 定义一个正则表达式模式，用于检测特殊情况（如缩写、电子邮件、百分比等）
    special_cases_pattern = re.compile(r'^(?:[A-Za-z]\.)+|\w+[@]\w+\.\w+|\d+%$|^(?:[\u4e00-\u9fff]+)$')
    # 如果当前 token 匹配到特殊情况（如 U.S.A., E-mail, 百分比，中文字符等），直接返回该 token
    if special_cases_pattern.match(token):
        return token

    # Strip unwanted punctuations from front and end # 去除 token 两端的标点符号
    token = token.strip(punctuations_to_strip)

    return token

def tokenize_and_filter(input_text: str) -> str:
    # 定义一个复杂的正则表达式模式，用于从文本中匹配不同的词汇、数字、标点等
    patterns = r"""(?x)  # Enable verbose mode, allowing regex to be on multiple lines and ignore whitespace # 启用详细模式，使正则表达式可以分行书写，忽略空白字符
                (?:[A-Za-z]\.)+          # Match abbreviations, e.g., U.S.A. # 匹配缩写词，例如 U.S.A.
                |\d+(?:\.\d+)?%?         # Match numbers, including percentages #匹配数字，可以包括小数和百分比
                |\w+(?:[-']\w+)*         # Match words, allowing for hyphens and apostrophes # 匹配单词，支持连字符和撇号（如 "mother-in-law", "I'm"）
                |(?:[\w\-\']@)+\w+       # Match email addresses # 匹配电子邮件地址
                """
    
    # 使用 re.findall 函数根据正则模式提取文本中的所有匹配项，返回匹配到的所有 token
    tokens = re.findall(patterns, input_text)

    # 停用词列表，用于过滤掉不需要的词
    stop_words = WORDS_TO_IGNORE

    filtered_tokens = []
    for token in tokens:
         # 将 token 转为小写，并进行清理（去除无效字符）
        token_lower = clean_en_token(token).lower()
        # 如果 token 不在停用词列表中，并且不是由纯标点符号组成，则将其添加到过滤后的 token 列表中
        if token_lower not in stop_words and not all(char in PUNCTUATIONS for char in token_lower):
            filtered_tokens.append(token_lower)

    return filtered_tokens

def string_tokenizer(text: str) -> List[str]:
    text = text.lower().strip() # 将文本转换为小写并去除前后空格
    if has_chinese_chars(text):
        import jieba
        _wordlist_tmp = list(jieba.lcut(text)) # 使用 jieba 进行分词，lcut 返回的是一个列表
        _wordlist = []
        for word in _wordlist_tmp:
            # 如果词汇不是由标点符号组成，则添加到 _wordlist 中
            if not all(char in PUNCTUATIONS for char in word):
                _wordlist.append(word)
    else:
        try:
            _wordlist = tokenize_and_filter(text)
        except Exception:
            logger.warning('Tokenize words by spaces.')
            _wordlist = text.split()
    _wordlist_res = []
    for word in _wordlist:
        if word in WORDS_TO_IGNORE:
            continue
        else:
            _wordlist_res.append(word)

    # 导入英文词干提取器（snowballstemmer）
    import snowballstemmer
    stemmer = snowballstemmer.stemmer('english')
    # 对过滤后的英文词汇进行词干化处理，并返回词干化后的结果
    return stemmer.stemWords(_wordlist_res)

def split_text_into_keywords(text: str) -> List[str]:
    # 使用 string_tokenizer 函数对文本进行分词，得到一个词汇列表
    _wordlist = string_tokenizer(text)
    wordlist = []
    for x in _wordlist:
        if x in WORDS_TO_IGNORE:
            continue
        wordlist.append(x)  # 如果当前词汇不是需要忽略的词，则添加到最终的关键词列表中
    return wordlist

'''
这段代码的核心任务是从提供的文本中提取关键词，并使用词干提取进行标准化，最终返回有效的关键词列表。
它还具有容错能力，在解析和提取过程中出现异常时，可以回退到简单的文本分割方法进行处理。
'''
def parse_keyword(text):
    try:
        res = json5.loads(text)
    except Exception:
        return split_text_into_keywords(text)

    import snowballstemmer
    stemmer = snowballstemmer.stemmer('english') # 创建一个英文的词干提取器

    # json format
    _wordlist = [] # 初始化一个空的列表，用于存储处理后的关键词
    try:
        # 如果 JSON 字典中存在 'keywords_zh' 且其值为列表类型，提取其中的中文关键词
        if 'keywords_zh' in res and isinstance(res['keywords_zh'], list):
            _wordlist.extend([kw.lower() for kw in res['keywords_zh']])
        
        # 如果 JSON 字典中存在 'keywords_en' 且其值为列表类型，提取其中的英文关键词
        if 'keywords_en' in res and isinstance(res['keywords_en'], list):
            _wordlist.extend([kw.lower() for kw in res['keywords_en']])
        
        # 使用 snowballstemmer 对关键词列表进行词干提取，将不同形式的词归一化为同一词根
        _wordlist = stemmer.stemWords(_wordlist)
        wordlist = [] # 创建一个空的列表用于存储最终的有效关键词
        # 遍历词干化后的所有关键词
        for x in _wordlist:
            if x in WORDS_TO_IGNORE:  # 如果当前关键词在需要忽略的词列表 `WORDS_TO_IGNORE` 中，则跳过
                continue
            wordlist.append(x)# 将有效的关键词添加到结果列表中
        
         # 使用 `split_text_into_keywords` 函数从 JSON 数据中的 'text' 字段提取更多的关键词
        split_wordlist = split_text_into_keywords(res['text'])
        wordlist += split_wordlist # 将从文本中提取的关键词添加到最终的关键词列表中
        return wordlist # 返回最终的关键词列表
    except Exception:
        # TODO: This catch is too broad.
        # 如果在处理过程中出现异常（例如，字典格式问题或缺少字段等），则返回通过 `split_text_into_keywords` 提取的关键词
        return split_text_into_keywords(text)