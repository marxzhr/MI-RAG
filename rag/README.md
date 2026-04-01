# LangChain RAG Baseline

当前目录提供一套可实验的基础 RAG baseline。

## 文件说明

- `data/kb_docs.jsonl`
  知识库文档，当前为示例医学知识库。
- `build_index.py`
  离线建库脚本，负责文档加载、切分、向量化和 FAISS 索引持久化。
- `run_baseline_rag.py`
  在线问答脚本，负责加载索引、检索 top-k 文档并调用大模型回答。
- `prepare_cmedqa2.py`
  将 `cMedQA2` 的压缩数据转换成 baseline 可用的知识库和评测集。
- `eval_baseline_rag.py`
  批量评测检索效果，输出 `Hit@1/3/5` 和 `MRR`。
- `prompt_template.py`
  独立 prompt 模板文件。
- `vector_store/`
  持久化后的 FAISS 索引目录。
- `run_rag_demo.py`
  之前的单文件样例版脚本，保留用于对照。

## 依赖

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters langchain-huggingface faiss-cpu python-dotenv sentence-transformers modelscope tqdm
```

## 环境变量

推荐在 `rag/.env` 里配置：

```bash
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=你的兼容接口地址
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=10
EMBEDDING_PROVIDER=openai
```

脚本启动时会自动读取 `rag/.env`。

如果你使用百炼 / Qwen 的 OpenAI 兼容 embedding 接口，建议显式设置：

```bash
EMBEDDING_BATCH_SIZE=10
```

如果你想改成本地 `bge-base-zh-v1.5` 做向量化，可以这样配：

```bash
EMBEDDING_PROVIDER=local_bge
LOCAL_EMBEDDING_MODEL=./models/BAAI/bge-base-zh-v1.5
EMBEDDING_DEVICE=cpu
OPENAI_API_KEY=你的Qwen或兼容API Key
OPENAI_BASE_URL=你的兼容接口地址
OPENAI_MODEL=qwen-plus
```

如果你的离线目录名是 `bge-base-zh-v1___5`，脚本现在会自动兼容解析；评测和问答脚本也会输出本地模型诊断信息，帮助确认模型来源和加载方式是否一致。

本地 embedding 模式下，检索向量不再走在线 API；只有最终生成回答还会调用聊天模型。

如果你不想用 `.env`，也可以继续用命令行环境变量：

```bash
set OPENAI_API_KEY=你的key
set OPENAI_BASE_URL=你的兼容接口地址
set OPENAI_MODEL=gpt-4o-mini
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## 构建索引

```bash
python rag\build_index.py
```

也可以指定数据源和索引目录：

```bash
python rag\build_index.py --data-path rag\data\cmedqa2_answers.jsonl --vector-store-dir rag\vector_store\cmedqa2
```

如果你想先做小规模 smoke test，并看到建库进度：

```bash
python rag\build_index.py --data-path rag\data\cmedqa2_answers.jsonl --vector-store-dir rag\vector_store\cmedqa2_small --limit-docs 5000 --batch-size 500
```

`build_index.py` 现在支持：

- `--limit-docs`: 只取前 N 条文档建索引
- `--batch-size`: 分批写入 FAISS 的 chunk 数
- `tqdm` 进度条：显示建库批次进度

## 问答

```bash
python rag\run_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --query "糖尿病怎么治疗？" --top-k 3
```

可选开启 rerank：

```bash
python rag\run_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --query "糖尿病怎么治疗？" --top-k 5 --fetch-k 20 --enable-dedup 1 --enable-rerank 1 --rerank-model BAAI/bge-reranker-base
```

按文档建议启用“实体提取 + 多路 Query Rewrite”：

```bash
python rag\run_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --query "我最近老是头晕是不是贫血啊怎么办" --top-k 5 --enable-query-understanding 1 --rewrite-count 3
```

默认检索会先召回 `top20`，再做两层去重：

- `doc_id` 去重
- 近重复文本去重

## 转换 cMedQA2

```bash
python rag\prepare_cmedqa2.py
```

会生成：

- `rag/data/cmedqa2_answers.jsonl`
- `rag/data/cmedqa2_dev_queries.jsonl`
- `rag/data/cmedqa2_test_queries.jsonl`

## 评测 baseline

```bash
python rag\eval_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --eval-path rag\data\cmedqa2_dev_queries.jsonl --top-k 5
```

评测 `Baseline vs Baseline + Entity + Multi-query Rewrite`：

```bash
python rag\eval_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --eval-path rag\data\cmedqa2_dev_queries.jsonl --top-k 5 --output-path rag\data\eval_no_rewrite.json
python rag\eval_baseline_rag.py --vector-store-dir rag\vector_store\cmedqa2_full --eval-path rag\data\cmedqa2_dev_queries.jsonl --top-k 5 --enable-query-understanding 1 --rewrite-count 3 --output-path rag\data\eval_with_multi_query.json
```

查询理解结果默认会缓存在：

- `rag/data/query_understanding_cache.json`

评测结果现在会额外保存：

- 原始召回池
- 去重统计
- 最终 top-k 文本与分数
- `gold_in_fetch_pool`，用于定位“召回到了但排序没排上来”

基于评测结果做失败样本抽样：

```bash
python rag\analyze_eval_results.py --input-path rag\data\eval_results.json --sample-size 100
```

做检索策略对比实验：

```bash
python rag\compare_retrieval_experiments.py --vector-store-dir rag\vector_store\cmedqa2_full --eval-path rag\data\cmedqa2_dev_queries.jsonl --top-k 5 --fetch-k 20 --rerank-model BAAI/bge-reranker-base
```

脚本会依次比较：

- baseline：无去重，无 rerank
- dedup：仅去重
- dedup_rerank：去重 + rerank

## 训练意图分类小模型

使用 `KUAKE-QIC` 训练 `ERNIE-Health` 文本分类器：

```bash
python rag\train_intent_classifier.py
```

如果显存较小，可以减小 batch size：

```bash
python rag\train_intent_classifier.py --batch-size 8 --eval-batch-size 16
```

当前默认推荐参数：

- `model_name=nghuyong/ernie-health-zh`
- `max_length=96`
- `epochs=5`
- `learning_rate=3e-5`
- `batch_size=16`
- `eval_batch_size=32`
- `warmup_ratio=0.06`

高显存版本示例：

```bash
python rag\train_intent_classifier.py --batch-size 16 --eval-batch-size 32 --max-length 96 --epochs 5 --learning-rate 3e-5
```

低显存版本示例：

```bash
python rag\train_intent_classifier.py --batch-size 8 --eval-batch-size 16 --gradient-accumulation-steps 2 --max-length 96 --epochs 5 --learning-rate 3e-5
```
