# Knowledge_preprocess

- 1:agent_knowledgebank_batch_test.py 是批量测试请求召回情况的。
- 2：bm25_knowledgebank_batch_test.py 是批量测试bm25召回情况的。<br>
其他两个是单次输入测试，查看召回内容的topk的。根据这情况可以调整知识库，将一些查不到的，放到能查找到的条目中去。保证召回率。这个配合 txt转换工具，提取知识库的请求。然后使用过这些请求来批量测试召回率，可以做到资料中的内容召回提高到90%以上
