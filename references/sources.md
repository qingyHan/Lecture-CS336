# References --- 官方来源与补充阅读

调查日期：2026-10-27。下列链接用于追溯资料版本；课程资料可能更新，开课前应重新核对。

## A. 官方课程入口

1.  **Stanford CS336: Language Modeling from Scratch --- Spring 2026**
    -   https://cs336.stanford.edu/
    -   用途：当前课程版本、先修要求、19 次排课、作业目标、GPU
        资源提示与课程政策。
2.  **Spring 2025 archived course**
    -   https://cs336.stanford.edu/spring2025/
    -   用途：历史版本对照；不可自动视为 Spring 2026 的精确要求。
3.  **Spring 2024 archived course**
    -   https://cs336.stanford.edu/spring2024/
    -   用途：课程历史和旧版材料对照。

## B. 官方讲义、视频与可视化材料

1.  **Official lectures repository**
    -   https://github.com/stanford-cs336/lectures
    -   本次检查：当前 `main` 的 `lecture_01.py`，其中标注 Spring
        2026，并包含课程导论、LM
        landscape、课程主题、tokenization、character/byte/word tokenizer
        与 BPE 教学实现。
2.  **Lecture 1 source**
    -   https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
    -   关键位置：课程导论与版本说明；`tokenization()`；`CharacterTokenizer`、`ByteTokenizer`、`merge`、`BPETokenizer`、`train_bpe`。
3.  **Lecture recordings**
    -   当前课程主页的 Schedule / YouTube playlist
        链接：https://cs336.stanford.edu/
    -   官方 `lecture_01.py` 提到的 Spring 2025
        课程视频播放列表：https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT\_
    -   说明：2026
        课表有自己的录制入口；播放列表可访问性和是否逐讲发布应以官网当前链接为准。
4.  **Official tokenization interactive reference mentioned in lecture**
    -   https://tiktokenizer.vercel.app/?encoder=gpt2
    -   用途：直观看到不同字符串如何分词；输出受具体 tokenizer
        版本影响。

## C. 官方作业仓库

1.  **Assignment 1 --- Basics**
    -   https://github.com/stanford-cs336/assignment1-basics
    -   官方课程主页所链接的公开仓库；当前 README 标注 Spring
        2025，而仓库 main 最近在 2026 年仍有更新。使用时需核对 handout
        与当届要求，不能仅凭链接假设是 Spring 2026 完全同版。
2.  **Assignment 2 --- Systems**
    -   https://github.com/stanford-cs336/assignment2-systems
    -   当前 README 标注 CS336 Spring 2026；包含 A1 staff implementation
        的复用入口与 `cs336_systems` 的实现位置。目标包括
        profiling/benchmarking、Triton 注意力
        kernel、内存高效分布式训练。
3.  **Assignment 3 --- Scaling**
    -   https://github.com/stanford-cs336/assignment3-scaling
    -   当前 README/handout 标注 Spring 2026；包括 scaling
        相关实现和训练 API 入口。运行前检查 API
        访问方式、身份凭据和使用限制。
4.  **Assignment 4 --- Data**
    -   https://github.com/stanford-cs336/assignment4-data
    -   当前 README/handout 标注 Spring 2026；包括 Common Crawl
        数据处理、过滤/去重及训练数据准备。仓库说明最终训练配置使用 8 张
        B200 GPU。
5.  **Assignment 5 --- Alignment**
    -   https://github.com/stanford-cs336/assignment5-alignment
    -   当前 handout 标注 Spring 2026；仓库包含 GRPO 测试，并提供可选
        safety alignment / instruction tuning / RLHF 补充 handout。
6.  **Official course GitHub organization**
    -   https://github.com/stanford-cs336
    -   用途：发现官方 lecture、assignment 和 leaderboard 仓库。

## D. 第一讲核心阅读：Tokenization

1.  **Gage, P. (1994). A New Algorithm for Data Compression.**
    -   http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM
    -   用途：理解 Byte Pair Encoding 的数据压缩来源。
2.  **Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine
    Translation of Rare Words with Subword Units.**
    -   ACL Anthology：https://aclanthology.org/P16-1162/
    -   arXiv：https://arxiv.org/abs/1508.07909
    -   用途：BPE/subword segmentation 如何缓解固定词表的开放词汇问题。
3.  **Radford et al. (2019). Language Models are Unsupervised Multitask
    Learners.**
    -   https://openai.com/research/better-language-models
    -   用途：GPT-2 与 byte-level BPE 的历史关联；这不是当前 CS336
        的完整课程材料。
4.  **Xue et al. (2022). ByT5: Towards a Token-Free Future with
    Pre-trained Byte-to-Byte Models.**
    -   https://aclanthology.org/2022.tacl-1.17/
    -   用途：比较字节级建模的鲁棒性、词表简化和序列变长代价。
5.  **Yu et al. (2023). MEGABYTE: Predicting Million-byte Sequences with
    Multiscale Transformers.**
    -   https://arxiv.org/abs/2305.07185
    -   用途：研究多尺度架构如何降低直接按字节建模的成本。
6.  **Pagnoni et al. (2025). Byte Latent Transformer: Patches Scale
    Better Than Tokens.**
    -   https://aclanthology.org/2025.acl-long.453/
    -   用途：了解动态 byte patches 与 token-free
        语言模型的现代研究方向。

## E. 资料状态与可信度约定

-   **已核对**：本次打开并检查过课程主页、课表、第一讲源码及上述作业仓库主页/README
    中明确列出的版本信息。
-   **待逐讲核对**：Lecture 2--17 的每份 PDF/代码、完整作业 handout
    的每个小问、每个实验的具体参数与提交格式。会在对应课程开始前补充文件路径、重要要求与来源。
-   **不推断**：Lecture 18/19
    的官方课表只列出嘉宾姓名，没有列出讲座主题，因此本项目不会预先编造主题。
-   **不虚构实验**：本项目自己的实验结果必须来自实际运行；未运行的结果会标记为预期或待验证。


## F. Spring 2026 官方讲义仓库与版本

- **Stanford CS336 官方课程主页** — https://cs336.stanford.edu/
  - 本项目当前版本基准；课程 schedule 包含 19 个排课条目，其中 Lecture 18 为 guest lecture，但官方可核验材料存在缺口。
- **官方讲义仓库** — https://github.com/stanford-cs336/lectures
  - 本次核对的 main revision：de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e
  - 可执行讲义：lecture_01.py、lecture_02.py、lecture_06.py、lecture_07.py、lecture_10.py、lecture_12.py、lecture_13.py、lecture_14.py、lecture_17.py。
  - PDF 讲义：lecture_03.pdf、lecture_04.pdf、lecture_05.pdf、lecture_08.pdf、lecture_09.pdf、lecture_11.pdf、lecture_15.pdf、lecture_16.pdf。
  - 官方仓库没有找到 lecture_18.py/PDF/trace；Lecture 19 也没有独立 lecture PDF/trace，公开录像链接另记于对应笔记。

## G. 官方作业仓库与审计版本

| 作业 | 官方仓库 | 本次核对的 main commit |
|---|---|---|
| A1 Basics | https://github.com/stanford-cs336/assignment1-basics | a158843b20107949f1a8d7df1b05cd33b9166712 |
| A2 Systems | https://github.com/stanford-cs336/assignment2-systems | ca8bc81a59b70516f7ebb2da4808daade877c736 |
| A3 Scaling | https://github.com/stanford-cs336/assignment3-scaling | 03e9372992e913061b9e78b5cfcb62ad8a87de35 |
| A4 Data | https://github.com/stanford-cs336/assignment4-data | 0555bea66369872d912652debf10b115ca0688c8 |
| A5 Alignment | https://github.com/stanford-cs336/assignment5-alignment | c2734a26308710949fe13226960a1e8cece94b7e |

- A1–A4 当前本地作业文件的 blob SHA 与对应官方 main 文件完全一致。
- A5 非 CSV 文件均与官方版本匹配；45 个 CSV 的 blob SHA 不同，抽样内容在统一 CRLF/LF 后相同。完整大文件的逐字节归一化审计仍有边界，详见 [assignments/ASSIGNMENT_AUDIT.md](../assignments/ASSIGNMENT_AUDIT.md)。
- A1 上游 README 仍写 Spring 2025；当前课程主页仍链接该仓库，且 handout PDF SHA 匹配官方当前文件。保留上游原文并单独记录此版本标签差异。

## H. 第三方辅助资料（非官方）

- Lecture 9 scaling law 辅助笔记：https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture09-scaling-laws/
- Lecture 11 scaling case studies：https://github.com/flyfei-cmd/video-to-chapter-book/blob/main/chapter-books/stanford-cs336/lecture-11.md
- Lecture 15 mid/post-training：https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture15-mid-post-training/
- Lecture 16 RLVR：https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture16-rlvr/
- Lecture 18 Daniel Selsam background guide：https://daxd.top/courses/cs336/lessons/0018-guest-daniel-selsam.html
- Lecture 19 Dan Fu public notes：https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/guest-lecture-dan-fu/

第三方资料只用于辅助定位和补充理解，不代替官方来源。对于官方没有提供可核验材料的内容，笔记必须明确记录资料边界，不把推断写成官方观点。
