# COURSE_INDEX --- 课程资料索引

## 官方课程与本地资料的映射

  -----------------------------------------------------------------------------------------------------------------------
  官方讲次       官方主题            本地讲义                                 本地笔记                     对应作业
  -------------- ------------------- ---------------------------------------- ---------------------------- --------------
  1              Overview,           `lectures/01_overview_tokenization.md`   `notes/01_tokenization.md`   A1
                 tokenization                                                                              

  2              PyTorch (einops),   待创建                                   待创建                       A1 / A2
                 resource accounting                                                                       

  3              Architectures,      待创建                                   待创建                       A1
                 hyperparameters                                                                           

  4              Attention           待创建                                   待创建                       A1 / A3
                 alternatives and                                                                          
                 mixture of experts                                                                        

  5              GPUs, TPUs          待创建                                   待创建                       A2

  6              Kernels, Triton     待创建                                   待创建                       A2

  7              Parallelism         待创建                                   待创建                       A2

  8              Parallelism         待创建                                   待创建                       A2

  9              Scaling laws        待创建                                   待创建                       A3

  10             Inference           待创建                                   待创建                       A2 / A3

  11             Scaling laws        待创建                                   待创建                       A3

  12             Evaluation          待创建                                   待创建                       A3 / A4

  13             Data: sources,      待创建                                   待创建                       A4
                 datasets                                                                                  

  14             Data filtering,     待创建                                   待创建                       A4
                 deduplication,                                                                            
                 mixing, synthetic                                                                         
                 data                                                                                      

  15             Mid/post-training   待创建                                   待创建                       A5
                 (SFT/RLHF)                                                                                

  16             Post-training ---   待创建                                   待创建                       A5
                 RLVR                                                                                      

  17             Alignment ---       待创建                                   待创建                       A5 / 扩展
                 multimodality                                                                             

  18             Guest lecture:      资料核实后创建                           资料核实后创建               ---
                 Daniel Selsam                                                                             

  19             Guest lecture: Dan  资料核实后创建                           资料核实后创建               ---
                 Fu                                                                                        
  -----------------------------------------------------------------------------------------------------------------------

## 本地实践代码

-   `code/tokenization_demo.py`：字符 tokenizer、UTF-8 字节 tokenizer
    和教学用 byte-level BPE。
-   `tests/test_tokenization_demo.py`：round-trip、BPE 合并和 Unicode
    测试。
-   `experiments/README.md`：实验记录规范。

## 版本控制原则

官方课程链接是版本入口；本地笔记是原创教学材料。每一讲开课前会把官方源文件、文件名/链接、版本线索和重要差异写入
`references/sources.md`，再据此更新本地讲义。
