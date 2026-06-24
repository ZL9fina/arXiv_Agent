# Paper Scout Agent

面向科研论文学习的 AI Agent 项目。默认采用 **按需实时检索 arXiv + LLM 生成学习建议** 的架构，避免为了公共论文库长期维护本地 RAG、重复下载 PDF 和反复 embedding。

项目仍保留本地 RAG 模式，适合离线使用、私有资料、团队内部文档或需要稳定复现实验的论文集合。

## 架构选择

### 默认：实时检索模式

公共论文不是私有数据，全量保存成 RAG 库会带来几个问题：

- 需要持续更新，论文库越大维护成本越高。
- 新论文出现后需要重新抓取、切块、embedding。
- arXiv 本身已经是公开、可检索的数据源，没有必要默认复制一份大缓存。
- 用户需求通常是临时学习某个方向，更适合按需检索。

因此默认流程改为：

```text
用户输入学习目标
        ↓
LLM 将目标改写成 arXiv 检索式
        ↓
通过 arXiv 工具实时获取相关论文
        ↓
发现论文中的源码链接
        ↓
LLM 基于论文元数据生成阅读顺序、推荐理由、作者线索和复现建议
```

### 可选：本地 RAG 模式

本地 RAG 模式仍然存在，用于：

- 私有论文、内部资料、课程资料。
- 需要离线访问的论文集合。
- 想固定一个实验数据快照，便于复现推荐结果。

```text
arXiv / PDF / 私有文档
        ↓
文本解析与切块
        ↓
Embedding
        ↓
ChromaDB + SQLite
        ↓
本地 RAG 检索
```

## 功能

- 实时从 arXiv 检索论文元数据。
- 使用 LLM 将用户学习目标转换为更适合 arXiv 的英文检索式。
- 基于候选论文生成中文学习建议、阅读顺序、推荐理由、作者追溯线索和复现建议。
- 自动发现 GitHub / GitLab / Bitbucket / Papers with Code 链接。
- 在用户确认后下载源码仓库，并生成部署建议。
- 可选运行 MCP Server，将论文检索能力暴露给支持 MCP 的 Agent 客户端。
- 保留本地 RAG 构建能力：PDF 解析、文本切块、ChromaDB 向量库、SQLite 元数据。
- APScheduler 支持本地 RAG 库的月度自动更新。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config.example.yaml config.yaml
```

配置 LLM API Key：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
```

如果使用其他 OpenAI-compatible 服务，例如 DeepSeek、通义千问、Kimi 等，修改 `config.yaml`：

```yaml
llm:
  enabled: true
  provider: openai-compatible
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  model: gpt-4.1-mini
```

把 `base_url`、`api_key_env`、`model` 换成你的服务商配置即可。

## 实时论文学习

默认 `learn` 使用实时检索模式：

```powershell
python -m paper_agent.cli learn "我想学习 AI Agent 与 RAG 的结合" --config config.yaml --top-k 8
```

如果暂时没有配置 LLM，也可以跳过 LLM 生成，只输出实时检索到的论文列表：

```powershell
python -m paper_agent.cli learn "AI Agent RAG" --config config.yaml --top-k 8 --no-llm
```

## MCP Server

项目提供一个可选 MCP Server，把论文检索作为工具暴露出去：

```powershell
python -m paper_agent.cli mcp-server --config config.yaml
```

暴露的工具：

- `search_arxiv_papers(query, max_results)`: 实时检索 arXiv，并返回论文元数据和源码链接。
- `recommend_papers(topic, max_results, use_llm)`: 根据学习目标规划检索式、获取论文，并可调用 LLM 生成推荐报告。

这种方式更符合 Agent 开发思路：Agent 不需要维护公共论文 RAG 库，而是在需要时通过 MCP 工具获取最新论文上下文。

## 本地 RAG 模式

初始化数据库：

```powershell
python -m paper_agent.cli init --config config.yaml
```

手动更新本地 RAG 库：

```powershell
python -m paper_agent.cli update --config config.yaml --query "retrieval augmented generation" --max-results 20
```

使用本地 RAG 推荐论文：

```powershell
python -m paper_agent.cli learn "我想系统学习多模态 RAG" --config config.yaml --top-k 8 --mode local-rag
```

按作者追溯本地库：

```powershell
python -m paper_agent.cli authors "Yann LeCun" --config config.yaml
```

启动本地 RAG 月度更新：

```powershell
python -m paper_agent.cli scheduler --config config.yaml
```

## 配置说明

- `llm.base_url`: OpenAI-compatible Chat Completions API 地址。
- `llm.api_key_env`: 存放 API Key 的环境变量名。
- `llm.model`: 使用的大模型名称。
- `arxiv.default_queries`: 本地 RAG 月度更新时使用的默认检索主题。
- `arxiv.max_results_per_query`: 每个主题最多抓取多少篇。
- `ingestion.download_pdfs`: 本地 RAG 模式是否下载 PDF 并抽取全文。
- `github.token_env`: GitHub Search API token 的环境变量名。没有 token 时仍会从论文文本中提取源码链接。
- `scheduler.day/hour/minute`: 本地 RAG 月度更新时间。

## 数据目录

默认数据会写入 `data/`：

- `data/papers.sqlite3`: 论文、作者、源码仓库元数据。
- `data/chroma/`: Chroma 向量库。
- `data/pdfs/`: 下载的论文 PDF。
- `data/repos/`: 用户确认下载的源码仓库。

## 安全提示

陌生论文源码可能包含任意脚本。当前实现只在用户确认后克隆仓库，并生成部署建议；它不会自动安装依赖或运行服务。真正部署前建议先阅读 `DEPLOYMENT_NOTES.md`，再在隔离环境中执行。

