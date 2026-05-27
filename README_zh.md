<h1 align="center">📚 基础模型的无监督后训练</h1>
<h3 align="center">配套清单 · 综述</h3>

<p align="center">
  <em>综述的审计追溯配套</em><br>
  <strong><em>《基础模型的无监督后训练:综述》</em></strong>
  <br><sub>EMNLP 2026 投稿</sub>
</p>

<p align="center">
  <a href="#-严格-upt-方法"><img src="https://img.shields.io/badge/严格_UPT-80_个方法-2ea44f?style=flat-square" alt="严格 UPT 数"></a>
  <a href="#-相邻方法8"><img src="https://img.shields.io/badge/相邻-8_个方法-orange?style=flat-square" alt="相邻数"></a>
  <a href="#-仅文中提及6"><img src="https://img.shields.io/badge/仅文中提及-6_个方法-lightgrey?style=flat-square" alt="仅文中提及"></a>
  <a href="read/"><img src="https://img.shields.io/badge/单篇笔记-91_篇-blue?style=flat-square" alt="笔记数"></a>
  <a href="pdfs/"><img src="https://img.shields.io/badge/PDF-91_篇-blue?style=flat-square" alt="PDF 数"></a>
  <br>
  <a href="#-许可"><img src="https://img.shields.io/badge/笔记许可-CC--BY_4.0-yellow?style=flat-square" alt="许可"></a>
  <img src="https://img.shields.io/badge/覆盖范围-2023年1月_→_2026年5月-blueviolet?style=flat-square" alt="覆盖窗口">
  <img src="https://img.shields.io/badge/出处-EMNLP_2026_投稿-red?style=flat-square" alt="出处">
  <img src="https://img.shields.io/badge/状态-持续维护-success?style=flat-square" alt="状态">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
</p>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

---

## ✨ 概要

本仓库是一份精心整理、可供审计的清单,涵盖本综述判定为**严格无监督后训练（UPT）**的所有基础模型论文 —— *同时*透明记录所有我们考虑过的相邻案例与边界情形。

UPT = 在预训练之后进行的过程,**更新模型的参数 / 适配器 / 记忆**,且仅使用**无标注输入和同模型谱系的信号**,**不依赖任何外部监督**(无人工标注、标准答案、验证器反馈、工具执行结果或更强教师蒸馏)。

本配套清单可用于:
- 🔎 浏览全部 80 个严格 UPT 方法,按其**内部更新对象**分为 4 个 family。
- 🧭 明确查看每篇相邻论文被剔除的*确切理由*(哪个边界检查未通过)。
- 📖 在 `read/<bibkey>.md`(英文)或 `read_cn/<bibkey>.md`(中文)中阅读任一被引论文的 1–2 页论证依据,对应 PDF 位于 `pdfs/<bibkey>.pdf`。
- 📊 将行数与主论文中的表格进行交叉核对。

<p align="center">
  <img src="assets/taxonomy_overview.png" alt="严格 UPT 方法的更新对象分类树" width="92%">
  <br>
  <sub><em>按更新对象分类的体系。四个严格 UPT family 加上虚线表示的相邻方法分支。桥接案例用 <code>‡</code> 标记。复现自主论文附录图 A1。</em></sub>
</p>

## 📑 目录

- [概要](#-概要)
- [一览覆盖范围](#-一览覆盖范围)
- [边界检查 (B1–B4)](#-边界检查-b1b4)
- [严格 UPT 方法](#-严格-upt-方法)
  - [Family I — 预测统计优化](#-family-i--预测统计优化-25)
  - [Family I/IV 桥接](#-family-iiv-桥接-1)
  - [Family II — 样本关系监督](#-family-ii--样本关系监督-22)
  - [Family III — 自生成目标自举](#-family-iii--自生成目标自举-22)
  - [Family IV — 内部评估器自举](#-family-iv--内部评估器自举-10)
- [相邻方法(8)](#-相邻方法8)
- [仅文中提及(6)](#-仅文中提及6)
- [适应时序视角](#-适应时序视角)
- [仓库结构](#-仓库结构)
- [字段说明](#-字段说明)
- [筛选规程](#-筛选规程)
- [与主论文的行数对账](#-与主论文的行数对账)
- [边界审计要点](#-边界审计要点)
- [贡献与持续维护策略](#-贡献与持续维护策略)
- [许可](#-许可)

## 📊 一览覆盖范围

| 类别 | 数量 | 说明 |
|------|-----:|------|
| **严格 UPT** | **80** | 通过全部四项边界检查 B1–B4 |
| ↳ Family I(预测统计优化) | 25 + 1 ‡ 桥接 | NLL / 熵 / 置信度 / 几何统计量 |
| ↳ Family II(样本关系监督) | 22 | 多样本共识 / 多数投票 / 聚类 |
| ↳ Family III(自生成目标自举) | 22 | 自构造指令 / 论证依据 / 偏好对 |
| ↳ Family IV(内部评估器自举) | 10 | 自评判 / 元评判 / 评估器驱动 RL |
| **相邻** | 8 | 至少一项 B1–B4 未通过;保留以保证可追溯 |
| **仅文中提及** | 6 | 仅在论文正文或 forest 树图中出现 |
| **单篇笔记**(`read/`,英文) | 91 | 每篇被引论文一份论证依据 |
| **单篇笔记**(`read_cn/`,中文) | 86 | 与英文笔记对应;5 篇英文原稿无对应中文版 |
| **PDF**(`pdfs/`) | 91 | 每篇被引论文一份 PDF |

## ✅ 边界检查 (B1–B4)

一种方法被认定为**严格 UPT**,当且仅当下面四条*全部*成立(逐字摘自主论文 §2):

| 编号 | 检查 | 通俗解释 |
|------|------|---------|
| **B1** | 显式更新 | 该方法对参数 / 适配器 / 记忆 / 持久局部状态进行更新。 |
| **B2** | 内部信号 | 更新信号仅由无标注输入和同模型谱系样本或判断计算得到。 |
| **B3** | 无外部监督 | 不使用标准答案、验证器反馈、工具/代码执行结论、人工标注或更强教师标签。 |
| **B4** | 内部评估器 | 更新中所用的任何评判器 / 打分器 / 奖励模型也来自同一模型谱系。 |

未通过任一项检查的方法被路由到**相邻方法**(保留在清单中,*不*删除),并记录未通过的检查项。

## 🧩 严格 UPT 方法

> 每行链接到 1–2 页论证依据(`read/<bibkey>.md` 英文,以及对应的 `read_cn/<bibkey>.md` 中文版,若存在)和论文 PDF(`pdfs/<bibkey>.pdf`)。表格与主论文表 1–4 行行对应。

### 🟦 Family I — 预测统计优化 (25)

**更新对象:**模型给出的单观测标量 —— token NLL、序列似然、熵、置信度或其他几何 / 规则类统计量。

<details open>
<summary><strong>方法列表(点击折叠)</strong></summary>

| 方法 | 出处 | 更新对象 | 英文笔记 | 中文笔记 | PDF |
|------|------|----------|----------|----------|-----|
| **CPT-LM** | arXiv 2023 | parameters | [阅读](read/ke2023cptlm.md) | [阅读](read_cn/ke2023cptlm.md) | [pdf](pdfs/ke2023cptlm.pdf) |
| **Simple CPT** | arXiv 2024 | parameters | [阅读](read/qian2024simplescalable.md) | [阅读](read_cn/qian2024simplescalable.md) | [pdf](pdfs/qian2024simplescalable.pdf) |
| **LangAdapt CPT** | ACL 2025 | parameters | [阅读](read/elhady2025languageadapt.md) | [阅读](read_cn/elhady2025languageadapt.md) | [pdf](pdfs/elhady2025languageadapt.pdf) |
| **Stability-Gap CPT** | ACL 2025 | parameters | [阅读](read/guo2025stabilitygap.md) | [阅读](read_cn/guo2025stabilitygap.md) | [pdf](pdfs/guo2025stabilitygap.pdf) |
| **ReplayAlign CPT** | NeurIPS 2025 Workshop | parameters | [阅读](read/abbes2025replayalign.md) | [阅读](read_cn/abbes2025replayalign.md) | [pdf](pdfs/abbes2025replayalign.pdf) |
| **E2-LLM** | ACL Findings 2024 | parameters | [阅读](read/liu2024e2llm.md) | [阅读](read_cn/liu2024e2llm.md) | [pdf](pdfs/liu2024e2llm.pdf) |
| **Data Eng 128K** | arXiv 2024 | parameters | [阅读](read/fu2024dataengineering.md) | [阅读](read_cn/fu2024dataengineering.md) | [pdf](pdfs/fu2024dataengineering.pdf) |
| **LongContext Scaling** | arXiv 2023 | parameters | [阅读](read/xiong2024longcontext.md) | [阅读](read_cn/xiong2024longcontext.md) | [pdf](pdfs/xiong2024longcontext.pdf) |
| **TLM** | ICML 2025 | parameters | [阅读](read/hu2025test.md) | [阅读](read_cn/hu2025test.md) | [pdf](pdfs/hu2025test.pdf) |
| **TTT-NN** | ICLR 2024 | parameters | [阅读](read/jang2024tttnn.md) | [阅读](read_cn/jang2024tttnn.md) | [pdf](pdfs/jang2024tttnn.pdf) |
| **Long TTT** | arXiv 2025 | parameters | [阅读](read/bansal2026qttt.md) | [阅读](read_cn/bansal2026qttt.md) | [pdf](pdfs/bansal2026qttt.pdf) |
| **In-Place TTT** | ICLR 2026 | parameters | [阅读](read/feng2026inplace.md) | [阅读](read_cn/feng2026inplace.md) | [pdf](pdfs/feng2026inplace.pdf) |
| **EM-FT** | NeurIPS 2025 | parameters | [阅读](read/agarwal2025unreasonable.md) | [阅读](read_cn/agarwal2025unreasonable.md) | [pdf](pdfs/agarwal2025unreasonable.pdf) |
| **One-shot EM** | arXiv 2025 | parameters | [阅读](read/gao2025oneshot.md) | [阅读](read_cn/gao2025oneshot.md) | [pdf](pdfs/gao2025oneshot.pdf) |
| **EM-RL(seq)** | NeurIPS 2025 | parameters | [阅读](read/agarwal2025unreasonable.md) | [阅读](read_cn/agarwal2025unreasonable.md) | [pdf](pdfs/agarwal2025unreasonable.pdf) |
| **EM-RL(tok)** | NeurIPS 2025 | parameters | [阅读](read/agarwal2025unreasonable.md) | [阅读](read_cn/agarwal2025unreasonable.md) | [pdf](pdfs/agarwal2025unreasonable.pdf) |
| **RENT** | arXiv 2025 | parameters | [阅读](read/prabhudesai2025rent.md) | [阅读](read_cn/prabhudesai2025rent.md) | [pdf](pdfs/prabhudesai2025rent.pdf) |
| **RLSC** | arXiv 2025 | parameters | [阅读](read/li2025rlsc.md) | [阅读](read_cn/li2025rlsc.md) | [pdf](pdfs/li2025rlsc.pdf) |
| **SLOT** | arXiv 2025 | sample-local state | [阅读](read/hu2025slot.md) | [阅读](read_cn/hu2025slot.md) | [pdf](pdfs/hu2025slot.pdf) |
| **SyTTA** | arXiv 2025 | sample-local state (LoRA) | [阅读](read/xu2025you.md) | [阅读](read_cn/xu2025you.md) | [pdf](pdfs/xu2025you.pdf) |
| **Model Whisper** | arXiv 2025 | sample-local state (steering) | [阅读](read/kang2025modelwhisper.md) | [阅读](read_cn/kang2025modelwhisper.md) | [pdf](pdfs/kang2025modelwhisper.pdf) |
| **ULDTTA** | arXiv 2026 | sample-local state (layer-wise) | [阅读](read/xu2026uldtta.md) | [阅读](read_cn/xu2026uldtta.md) | [pdf](pdfs/xu2026uldtta.pdf) |
| **VIGOR** | arXiv 2026 | parameters | [阅读](read/wen2026verifierfreerlllmsintrinsic.md) | [阅读](read_cn/wen2026verifierfreerlllmsintrinsic.md) | [pdf](pdfs/wen2026verifierfreerlllmsintrinsic.pdf) |
| **Latent-GRPO** | arXiv 2026 | parameters | [阅读](read/zhang2026silencejudgereinforcementlearning.md) | [阅读](read_cn/zhang2026silencejudgereinforcementlearning.md) | [pdf](pdfs/zhang2026silencejudgereinforcementlearning.pdf) |
| **SSL-R1** | arXiv 2026 | parameters | [阅读](read/xie2026sslr1selfsupervisedvisualreinforcement.md) | [阅读](read_cn/xie2026sslr1selfsupervisedvisualreinforcement.md) | [pdf](pdfs/xie2026sslr1selfsupervisedvisualreinforcement.pdf) |

</details>

### 🔄 Family I/IV 桥接 (1)

**桥接:**该方法主签名属于 Family I,但带有次级的内部评估器特征;主论文表中以 `‡` 标记。

| 方法 | 出处 | 更新对象 | 英文笔记 | 中文笔记 | PDF |
|------|------|----------|----------|----------|-----|
| **SUDER** | arXiv 2025 | parameters | [阅读](read/hong2025suderselfimprovingunifiedlarge.md) | [阅读](read_cn/hong2025suderselfimprovingunifiedlarge.md) | [pdf](pdfs/hong2025suderselfimprovingunifiedlarge.pdf) |

### 🟩 Family II — 样本关系监督 (22)

**更新对象:**跨多个内部样本的某种关系 —— 多数投票、语义聚类、自置信度、路径一致性或成对一致。

<details open>
<summary><strong>方法列表(点击折叠)</strong></summary>

| 方法 | 出处 | 更新对象 | 英文笔记 | 中文笔记 | PDF |
|------|------|----------|----------|----------|-----|
| **EMPO** | arXiv 2025 | parameters | [阅读](read/zhang2025rightquestion.md) | [阅读](read_cn/zhang2025rightquestion.md) | [pdf](pdfs/zhang2025rightquestion.pdf) |
| **Intuitor** | ICLR 2026 | parameters | [阅读](read/zhao2025intuitor.md) | [阅读](read_cn/zhao2025intuitor.md) | [pdf](pdfs/zhao2025intuitor.pdf) |
| **CoVo** | arXiv 2025 | parameters | [阅读](read/zhang2025covo.md) | [阅读](read_cn/zhang2025covo.md) | [pdf](pdfs/zhang2025covo.pdf) |
| **Co-rewarding** | ICLR 2026 | parameters | [阅读](read/zhang2026corewarding.md) | [阅读](read_cn/zhang2026corewarding.md) | [pdf](pdfs/zhang2026corewarding.pdf) |
| **EvoLMM** | arXiv 2025 | parameters | [阅读](read/thawakar2025evolmm.md) | [阅读](read_cn/thawakar2025evolmm.md) | [pdf](pdfs/thawakar2025evolmm.pdf) |
| **TTRL** | NeurIPS 2025 | parameters | [阅读](read/zuo2025ttrl.md) | [阅读](read_cn/zuo2025ttrl.md) | [pdf](pdfs/zuo2025ttrl.pdf) |
| **ETTRL** | arXiv 2025 | parameters | [阅读](read/liu2025ettrl.md) | [阅读](read_cn/liu2025ettrl.md) | [pdf](pdfs/liu2025ettrl.pdf) |
| **ECHO** | arXiv 2026 | parameters | [阅读](read/zhao2026echo.md) | [阅读](read_cn/zhao2026echo.md) | [pdf](pdfs/zhao2026echo.pdf) |
| **SPINE** | arXiv 2025 | parameters | [阅读](read/wu2025spine.md) | [阅读](read_cn/wu2025spine.md) | [pdf](pdfs/wu2025spine.pdf) |
| **Self-Harmony** | ICLR 2026 | parameters | [阅读](read/wang2026selfharmony.md) | [阅读](read_cn/wang2026selfharmony.md) | [pdf](pdfs/wang2026selfharmony.pdf) |
| **DARE** | arXiv 2026 | parameters | [阅读](read/du2026dare.md) | [阅读](read_cn/du2026dare.md) | [pdf](pdfs/du2026dare.pdf) |
| **SCOPE** | arXiv 2025 | parameters | [阅读](read/wang2025scope.md) | [阅读](read_cn/wang2025scope.md) | [pdf](pdfs/wang2025scope.pdf) |
| **COMPASS** | arXiv 2025 | parameters | [阅读](read/xing2025compass.md) | [阅读](read_cn/xing2025compass.md) | [pdf](pdfs/xing2025compass.pdf) |
| **SCRL** | arXiv 2026 | parameters | [阅读](read/yan2026scrl.md) | [阅读](read_cn/yan2026scrl.md) | [pdf](pdfs/yan2026scrl.pdf) |
| **RLCCF** | arXiv 2025 | parameters | [阅读](read/yuan2025rlccf.md) | [阅读](read_cn/yuan2025rlccf.md) | [pdf](pdfs/yuan2025rlccf.pdf) |
| **RoiRL** | NeurIPS 2025 Workshop | parameters | [阅读](read/arzhantsev2025roirl.md) | [阅读](read_cn/arzhantsev2025roirl.md) | [pdf](pdfs/arzhantsev2025roirl.pdf) |
| **EVOL-RL** | arXiv 2025 | parameters | [阅读](read/zhou2025evolrl.md) | [阅读](read_cn/zhou2025evolrl.md) | [pdf](pdfs/zhou2025evolrl.pdf) |
| **TTRV** | arXiv 2025 | parameters | [阅读](read/singh2025ttrv.md) | [阅读](read_cn/singh2025ttrv.md) | [pdf](pdfs/singh2025ttrv.pdf) |
| **MM-UPT** | NeurIPS 2025 | parameters | [阅读](read/wei2025mmupt.md) | [阅读](read_cn/wei2025mmupt.md) | [pdf](pdfs/wei2025mmupt.pdf) |
| **Dual Consensus** | arXiv 2026 | parameters | [阅读](read/du2026dualconsensusescapingspurious.md) | [阅读](read_cn/du2026dualconsensusescapingspurious.md) | [pdf](pdfs/du2026dualconsensusescapingspurious.pdf) |
| **CSRS** | arXiv 2026 | parameters | [阅读](read/yu2026stabilizingunsupervisedselfevolutionmllms.md) | [阅读](read_cn/yu2026stabilizingunsupervisedselfevolutionmllms.md) | [pdf](pdfs/yu2026stabilizingunsupervisedselfevolutionmllms.pdf) |
| **EvoQuality** | ICLR 2026 | parameters | [阅读](read/wen2026selfevolvingvisionlanguagemodelsimage.md) | [阅读](read_cn/wen2026selfevolvingvisionlanguagemodelsimage.md) | [pdf](pdfs/wen2026selfevolvingvisionlanguagemodelsimage.pdf) |

</details>

### 🟨 Family III — 自生成目标自举 (22)

**更新对象:**模型为自己构造的合成目标 —— 指令、论证依据、课程、辩论轨迹或偏好对 —— 然后用 SFT / DPO 在其上训练。

<details open>
<summary><strong>方法列表(点击折叠)</strong></summary>

| 方法 | 出处 | 更新对象 | 英文笔记 | 中文笔记 | PDF |
|------|------|----------|----------|----------|-----|
| **Self-Tuning** | ACL Findings 2025 | parameters | [阅读](read/zhang2025selftuning.md) | [阅读](read_cn/zhang2025selftuning.md) | [pdf](pdfs/zhang2025selftuning.pdf) |
| **KBAlign** | EMNLP Findings 2025 | parameters | [阅读](read/zeng2025kbalign.md) | [阅读](read_cn/zeng2025kbalign.md) | [pdf](pdfs/zeng2025kbalign.pdf) |
| **CYCLE-INSTRUCT** | EMNLP 2025 | parameters | [阅读](read/shen2025cycleinstruct.md) | [阅读](read_cn/shen2025cycleinstruct.md) | [pdf](pdfs/shen2025cycleinstruct.pdf) |
| **Self-Improve** | EMNLP 2023 | parameters | [阅读](read/huang2023selfimprove.md) | [阅读](read_cn/huang2023selfimprove.md) | [pdf](pdfs/huang2023selfimprove.pdf) |
| **Quiet-STaR** | arXiv 2024 | parameters | [阅读](read/zelikman2024quietstar.md) | [阅读](read_cn/zelikman2024quietstar.md) | [pdf](pdfs/zelikman2024quietstar.pdf) |
| **Confident ST** | EMNLP Findings 2025 | parameters | [阅读](read/wang2025confidentreasoning.md) | [阅读](read_cn/wang2025confidentreasoning.md) | [pdf](pdfs/wang2025confidentreasoning.pdf) |
| **GENIUS** | arXiv 2025 | parameters | [阅读](read/xu2025genius.md) | [阅读](read_cn/xu2025genius.md) | [pdf](pdfs/xu2025genius.pdf) |
| **LRM Self-Train** | arXiv 2025 | parameters | [阅读](read/shi2025lrmselftrain.md) | [阅读](read_cn/shi2025lrmselftrain.md) | [pdf](pdfs/shi2025lrmselftrain.pdf) |
| **DTE** | EMNLP 2025 | parameters | [阅读](read/liu2025dte.md) | [阅读](read_cn/liu2025dte.md) | [pdf](pdfs/liu2025dte.pdf) |
| **LongMagpie** | arXiv 2025 | parameters | [阅读](read/gao2025longmagpie.md) | [阅读](read_cn/gao2025longmagpie.md) | [pdf](pdfs/gao2025longmagpie.pdf) |
| **Long Self-Improve** | arXiv 2024 | parameters | [阅读](read/wang2024longselfimprove.md) | [阅读](read_cn/wang2024longselfimprove.md) | [pdf](pdfs/wang2024longselfimprove.pdf) |
| **TTCS** | arXiv 2026 | parameters | [阅读](read/yang2026ttcs.md) | [阅读](read_cn/yang2026ttcs.md) | [pdf](pdfs/yang2026ttcs.pdf) |
| **DiSCTT** | arXiv 2026 | parameters | [阅读](read/moradi2026disctt.md) | [阅读](read_cn/moradi2026disctt.md) | [pdf](pdfs/moradi2026disctt.pdf) |
| **TTSR** | arXiv 2026 | parameters | [阅读](read/he2026ttsr.md) | [阅读](read_cn/he2026ttsr.md) | [pdf](pdfs/he2026ttsr.pdf) |
| **R-Zero** | arXiv 2025 | parameters | [阅读](read/huang2026rzero.md) | [阅读](read_cn/huang2026rzero.md) | [pdf](pdfs/huang2026rzero.pdf) |
| **ScPO** | ICML 2025 | parameters | [阅读](read/prasad2025scpo.md) | [阅读](read_cn/prasad2025scpo.md) | [pdf](pdfs/prasad2025scpo.pdf) |
| **MACA** | arXiv 2025 | parameters | [阅读](read/samanta2025maca.md) | [阅读](read_cn/samanta2025maca.md) | [pdf](pdfs/samanta2025maca.pdf) |
| **LongPO** | ICLR 2025 | parameters | [阅读](read/chen2025longpo.md) | [阅读](read_cn/chen2025longpo.md) | [pdf](pdfs/chen2025longpo.pdf) |
| **RLSF** | arXiv 2025 | parameters | [阅读](read/vanniekerk2025rlsf.md) | [阅读](read_cn/vanniekerk2025rlsf.md) | [pdf](pdfs/vanniekerk2025rlsf.pdf) |
| **G-Zero** | arXiv 2026 | parameters | [阅读](read/huang2026gzeroselfplayopenendedgeneration.md) | [阅读](read_cn/huang2026gzeroselfplayopenendedgeneration.md) | [pdf](pdfs/huang2026gzeroselfplayopenendedgeneration.pdf) |
| **QueST** | arXiv 2026 | sample-local state (LoRA) | [阅读](read/song2026queryconditionedtesttimeselftraininglarge.md) | [阅读](read_cn/song2026queryconditionedtesttimeselftraininglarge.md) | [pdf](pdfs/song2026queryconditionedtesttimeselftraininglarge.pdf) |
| **V-Zero** | arXiv 2026 | parameters | [阅读](read/wang2026vzeroselfimprovingmultimodalreasoning.md) | [阅读](read_cn/wang2026vzeroselfimprovingmultimodalreasoning.md) | [pdf](pdfs/wang2026vzeroselfimprovingmultimodalreasoning.pdf) |

</details>

### 🟥 Family IV — 内部评估器自举 (10)

**更新对象:**自抬升的*评判器*(打分器 / 奖励模型 / 元评判),由同一谱系生成和消费;actor 针对其裁定进行训练。

<details open>
<summary><strong>方法列表(点击折叠)</strong></summary>

| 方法 | 出处 | 更新对象 | 英文笔记 | 中文笔记 | PDF |
|------|------|----------|----------|----------|-----|
| **Self-Rewarding LM** | ICML 2024 | parameters | [阅读](read/yuan2024selfrewarding.md) | [阅读](read_cn/yuan2024selfrewarding.md) | [pdf](pdfs/yuan2024selfrewarding.pdf) |
| **CREAM** | ICLR 2025 | parameters | [阅读](read/wang2024cream.md) | [阅读](read_cn/wang2024cream.md) | [pdf](pdfs/wang2024cream.pdf) |
| **Meta-Rewarding** | arXiv 2024 | parameters | [阅读](read/wu2024metarewarding.md) | [阅读](read_cn/wu2024metarewarding.md) | [pdf](pdfs/wu2024metarewarding.pdf) |
| **Temporal SRLM** | arXiv 2025 | parameters | [阅读](read/jin2025temporalselfrewarding.md) | [阅读](read_cn/jin2025temporalselfrewarding.md) | [pdf](pdfs/jin2025temporalselfrewarding.pdf) |
| **CoNL** | arXiv 2026 | parameters | [阅读](read/sui2026conl.md) | [阅读](read_cn/sui2026conl.md) | [pdf](pdfs/sui2026conl.pdf) |
| **RLME** | arXiv 2026 | parameters | [阅读](read/rentschler2026rlme.md) | [阅读](read_cn/rentschler2026rlme.md) | [pdf](pdfs/rentschler2026rlme.pdf) |
| **Meta-TTRL** | arXiv 2026 | parameters | [阅读](read/tan2026metattrl.md) | [阅读](read_cn/tan2026metattrl.md) | [pdf](pdfs/tan2026metattrl.pdf) |
| **AERO** | arXiv 2026 | parameters | [阅读](read/gao2026aeroautonomousevolutionaryreasoning.md) | [阅读](read_cn/gao2026aeroautonomousevolutionaryreasoning.md) | [pdf](pdfs/gao2026aeroautonomousevolutionaryreasoning.pdf) |
| **Self-Judge** | arXiv 2026 | parameters | [阅读](read/wu2026modelsjudgethemselvesunsupervised.md) | [阅读](read_cn/wu2026modelsjudgethemselvesunsupervised.md) | [pdf](pdfs/wu2026modelsjudgethemselvesunsupervised.pdf) |
| **GvU** | CVPR 2026 | parameters | [阅读](read/pan2026learninggenerateunderstandingunderstandingdriven.md) | [阅读](read_cn/pan2026learninggenerateunderstandingunderstandingdriven.md) | [pdf](pdfs/pan2026learninggenerateunderstandingunderstandingdriven.pdf) |

</details>

## 🚧 相邻方法(8)

与 UPT 紧密相关但**未通过至少一项边界检查**。保留在清单中以保证可追溯。

| 方法 | 出处 | 未通过 | 相邻类型 | 英文笔记 | 中文笔记 | PDF |
|------|------|--------|----------|----------|----------|-----|
| **EM-INF** | NeurIPS 2025 | `B1` | 无更新推理时 | [阅读](read/agarwal2025unreasonable.md) | [阅读](read_cn/agarwal2025unreasonable.md) | [pdf](pdfs/agarwal2025unreasonable.pdf) |
| **T3RL** | arXiv 2026 | `B3` | 验证器/工具辅助 | [阅读](read/liao2026t3rl.md) | [阅读](read_cn/liao2026t3rl.md) | [pdf](pdfs/liao2026t3rl.pdf) |
| **Absolute Zero** | arXiv 2025 | `B3` | 验证器/工具辅助 | [阅读](read/zhao2025absolutezero.md) | [阅读](read_cn/zhao2025absolutezero.md) | [pdf](pdfs/zhao2025absolutezero.pdf) |
| **Concise ST** | ACL Findings 2025 | `B3` | 验证器/工具辅助 | [阅读](read/wang2025concisereasoning.md) | [阅读](read_cn/wang2025concisereasoning.md) | [pdf](pdfs/wang2025concisereasoning.pdf) |
| **LEPA** | arXiv 2025 | `B3` | 验证器/工具辅助 | [阅读](read/zhang2025lepa.md) | [阅读](read_cn/zhang2025lepa.md) | [pdf](pdfs/zhang2025lepa.pdf) |
| **Self-Instruct** | ACL 2023 | `B3` | 种子/人工监督 | [阅读](read/wang2023selfinstruct.md) | [阅读](read_cn/wang2023selfinstruct.md) | [pdf](pdfs/wang2023selfinstruct.pdf) |
| **Instruction-Backtranslation** | arXiv 2023 | `B3` | 种子/人工监督 | [阅读](read/li2024instructionbacktranslation.md) | [阅读](read_cn/li2024instructionbacktranslation.md) | [pdf](pdfs/li2024instructionbacktranslation.pdf) |
| **CSR** | arXiv 2024 | `B4` | 外部评估器 | [阅读](read/zhou2024csr.md) | [阅读](read_cn/zhou2024csr.md) | [pdf](pdfs/zhou2024csr.pdf) |

## 📝 仅文中提及(6)

仅在**章节正文或 forest 树图**中提及,未列入四个严格 UPT 表格的方法。单列以保持严格计数恰好为 80。

| 方法 | 主家族 | 提及位置 | 英文笔记 | 中文笔记 | PDF |
|------|--------|----------|----------|----------|-----|
| **PowerFlow** | F1 | §Family I prose (main.tex L796) + forest tree (L284) | [阅读](read/chen2026powerflowunlockingdualnature.md) | [阅读](read_cn/chen2026powerflowunlockingdualnature.md) | [pdf](pdfs/chen2026powerflowunlockingdualnature.pdf) |
| **Multi-Reward RLIF** | F2 | §Family II prose (main.tex L936) + forest tree (L303) | [阅读](read/joarder2026betteronecollapsefreemultireward.md) | [阅读](read_cn/joarder2026betteronecollapsefreemultireward.md) | [pdf](pdfs/joarder2026betteronecollapsefreemultireward.pdf) |
| **SePT** | F3 | §Family III prose (main.tex L1025) + forest tree (L320) | [阅读](read/li2026modelhelpitselfrewardfree.md) | [阅读](read_cn/li2026modelhelpitselfrewardfree.md) | [pdf](pdfs/li2026modelhelpitselfrewardfree.pdf) |
| **PonderTTT** | F1-timing | §7 Timing prose (main.tex L1215) + forest tree (L496) | [阅读](read/sim2026ponderttt.md) | [阅读](read_cn/sim2026ponderttt.md) | [pdf](pdfs/sim2026ponderttt.pdf) |
| **TT-VLA** | F1-timing | §7 Timing prose (main.tex L1206) + forest tree (L482) | [阅读](read/liu2026ttvla.md) | [阅读](read_cn/liu2026ttvla.md) | [pdf](pdfs/liu2026ttvla.pdf) |
| **SECL** | F1-timing | §7 Timing prose (main.tex L1206) + forest tree (L482) | [阅读](read/strich2026secl.md) | [阅读](read_cn/strich2026secl.md) | [pdf](pdfs/strich2026secl.pdf) |

## ⏱️ 适应时序视角

与四个 family 正交:每个方法独立沿**输入可见性 × 更新持久性**(主论文 §6)进行编码。下表计数之和 ≥ 80,因为部分论文按协议贡献多于一条。

| 时序 | 成员数 | 模式 |
|------|------:|------|
| **离线语料 UPT** | 59 | 部署前在无标注语料上训练 |
| **全目标集转导适应** | 11 | 先看到整个目标集,再更新,再推理 |
| **少样本目标适应** | 1 | 在少量切片上适应;泛化到留出集 |
| **流式持续适应** | 1 | 在线更新随流式目标持续累积 |
| **测试时实例适应** | 7 | 逐实例更新;在实例边界处重置 |
| **序列内适应** | 1 | 更新与生成在一个序列内交替进行 |
| *(相邻)* 无更新推理 | 1 | 推理时 logit / 状态编辑,无持久更新(EM-INF) |

## 📂 仓库结构

```
awesome-unsupervised-post-training/
├── README.md                    ← 英文版
├── README_zh.md                 ← 你正在阅读(中文)
├── strict_upt_methods.csv       ← 80 行,每行一个严格 UPT 方法
├── adjacent_methods.csv         ← 8 行,每行一个相邻方法(未通过 B1–B4 之一)
├── prose_only_methods.csv       ← 6 行,仅正文 / forest 树中提及
├── ambiguous_cases.md           ← 边界情形决策日志
├── assets/                      ← README 图片
├── read/<bibkey>.md             ← 91 篇方法论证依据(英文)
├── read_cn/<bibkey>.md          ← 86 篇方法论证依据(中文)
└── pdfs/<bibkey>.pdf            ← 91 篇 PDF
```

## 🗂️ 字段说明

<details>
<summary><strong><code>strict_upt_methods.csv</code></strong></summary>

| 列名 | 说明 |
|------|------|
| `family` | `F1` / `F2` / `F3` / `F4`(SUDER 为 `F1/IV-bridge`)。 |
| `method_label` | 与主论文表格完全一致的方法名。 |
| `bibkey` | 主论文参考文献中的 BibTeX 键。 |
| `update_target` | 参数 / 适配器 / 记忆 / 持久局部状态。 |
| `signal_source` | 梯度作用的内部更新对象。 |
| `B1` … `B4` | 边界检查裁定结果;`Y` / `N` / `N/A`。 |
| `timing_regime` | 取值之一:`offline_corpus`、`full_cohort_transductive`、`few_sample_target`、`streaming_continual`、`test_time_instance`、`within_sequence`。 |
| `venue` | 会议 / 期刊 / arXiv 预印本,并标注年份。 |
| `evidence_status` | `explicit` / `clarified` / `reverse_engineered`。 |
| `note_path` | `read/<bibkey>.md` 的路径。 |
| `rationale_short` | 一行摘要;完整论证依据见对应 `.md` 笔记。 |

</details>

<details>
<summary><strong><code>adjacent_methods.csv</code></strong></summary>

同上,另加:

- `failed_check` —— 方法未通过 B1–B4 中的哪一项。
- `adjacent_type` —— 取值之一:`no_update_inference_time`、`verifier_or_tool_assisted`、`seed_or_human_supervised`、`stronger_teacher_distillation`、`external_evaluator`。

</details>

<details>
<summary><strong><code>prose_only_methods.csv</code></strong></summary>

与 `strict_upt_methods.csv` 一致,另加 `mention_location`(方法在主论文中被提及的章节 + 行号,并标注是否出现在 forest 树图中)。

</details>

## 🔍 筛选规程

与主论文附录 A 一致。

- **覆盖窗口:**2023 年 1 月 – 2026 年 5 月;在无标注 prompts、文本或目标输入上进行预训练后更新的纯文本与多模态基础模型方法。
- **检索来源:**ACL Anthology · arXiv · Semantic Scholar · Google Scholar。
- **种子线索:**(i) 持续预训练 + 测试时训练 ·(ii)自我提升 + 自我奖励 ·(iii)基于内部共识的测试时 RL ·(iv)多模态 UPT。
- **纳入条件:**B1–B4 必须全部成立,候选方法须有可验证的算法描述,且须作用于基础模型规模的文本或多模态模型。
- **排除条件:**综述、纯基准论文、硬件 / 系统论文,以及与后训练贡献无关的特定领域应用。
- **相邻路由:**未通过任一检查的方法进入 `adjacent_methods.csv`,并记录未通过的检查项 —— *不*删除。

## 🧮 与主论文的行数对账

- `80 = 26 (F1) + 22 (F2) + 22 (F3) + 10 (F4)`,与 `strict_upt_methods.csv` 一致。
  - F1 的 26 由 25 个严格 F1 行 + 1 个 `F1/IV-bridge` 行(**SUDER**,主论文表 1 中标 `‡`)组成。
  - 桥接案例 **SUDER**(F1)与 **GvU**(F4)按更新对象规则,分别*只*算一次,归入其主家族。
- **CSR**(`zhou2024csr`)在 `adjacent_methods.csv` 中,`failed_check=B4`。它*不*在 `strict_upt_methods.csv` 中;主论文 Family-IV 的 caption 明确标注其"在 (B4) 下属于相邻"。
- 六个仅文中提及的方法迁入 `prose_only_methods.csv`,使严格计数恰好保持 80:**PowerFlow、Multi-Reward RLIF、SePT**(家族正文)+ **PonderTTT、TT-VLA、SECL**(§6 时序正文)。
- 主论文四个严格 UPT 表格中的每一行 `\textsc{…}~\citep{<bibkey>}` 在此恰好对应一条记录。

## 🛡️ 边界审计要点

**带有显式 caveat 的严格 UPT 行(80 中的 2 条):**

- `zhang2025rightquestion`(**EMPO**)—— `B4=N` caveat:自由形式变体使用了外部 General-Verifier 1.5B SLM;数学变体则无此问题。
- `yuan2025rlccf`(**RLCCF**)—— `B2/B4` caveat:跨群体异构 LLM,非严格同谱系。

其余 78 条严格 UPT 行皆为 `Y/Y/Y/{Y 或 N/A}` 干净通过。

**相邻行 —— 各方法未通过的检查项:**

| 相邻方法 | 未通过 | 原因 |
|----------|--------|------|
| **EM-INF** | `B1` | 无持久更新 —— 仅推理时的 logit 下降 |
| **T³RL** · **Absolute Zero** · **Concise ST** · **LEPA** | `B3` | 外部验证器 / 正确性过滤 / 工具执行 |
| **Self-Instruct** · **Instruction-Backtranslation** | `B3` | 从人工编写的种子样本启动 |
| **CSR** | `B4` | CLIP 推导的奖励项来自非同谱系编码器 |

<!--
## 📖 引用本工作

如果你使用本配套或综述,请引用:

```bibtex
@inproceedings{unsupervised_post_training_survey_2026,
  title     = {Unsupervised Post-Training of Foundation Models: A Survey},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026 (submission)},
  year      = {2026},
  note      = {Companion inventory: this repository.}
}
```
-->

## 🤝 贡献与持续维护策略

如主论文 *Limitations* 一节所述,本清单将随领域演进持续更新。每次冻结后新增一条都需要:

1. 在对应 CSV(`strict_upt_methods.csv` / `adjacent_methods.csv` / `prose_only_methods.csv`)中加一行。
2. 在 `read/<bibkey>.md` 中写一份 1–2 页论证依据,回答:*归入哪一 family?更新对象是什么?哪一边界检查会失败(若有)?*
3. 将论文 PDF 放入 `pdfs/<bibkey>.pdf`。
4. 若属于边界情形,在 `ambiguous_cases.md` 中加一条记录。

**提议新方法**的建议流程:

1. 提一个 issue,附上论文链接和一行 family 提案。
2. 我们将对其应用 B1–B4 边界检查,并记录路由决策。
3. 接受后,把 CSV 行 + `read/` 笔记 + PDF 作为单次变更提交。

## 📄 许可

- **配套笔记**(`read/`、`read_cn/`、三个 CSV、`ambiguous_cases.md`、本 README):以 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。
- **`pdfs/` 中所引用的 PDF**:仍受其原始出版方许可约束 —— 在此提供仅为便利读者及 EMNLP 2026 评审流程。

---

<p align="center"><sub><em>《基础模型的无监督后训练:综述》(EMNLP 2026 投稿)的配套 · 覆盖范围冻结于 2026 年 5 月 · 索引最近刷新:2026-05-27。</em></sub></p>
<p align="center"><a href="#-基础模型的无监督后训练">⬆ 返回顶部</a></p>
