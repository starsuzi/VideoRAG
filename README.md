# 📽️ VideoRAG: Retrieval-Augmented Generation over Video Corpus

[![Paper](https://img.shields.io/badge/arXiv-2501.05874-b31b1b)](https://arxiv.org/abs/2501.05874)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/downloads/release/python-380/)

🚀 **Welcome to the official repository of** [**VideoRAG: Retrieval-Augmented Generation over Video Corpus**](https://arxiv.org/abs/2501.05874)!

## 🔍 What is VideoRAG?

![VideoRAG](./assets/figure.png)

VideoRAG is a Retrieval-Augmented Generation (RAG) framework designed to bring video retrieval into the generative AI landscape. While most RAG systems focus on text retrieval, VideoRAG dynamically retrieves videos based on relevance and incorporates both visual and textual information into response generation.

Our VideoRAG implementation includes the following key features:
* ✅ **Dynamic Video Retrieval:** Selects relevant videos based on query relevance.
* ✅ **Multimodal Representation:** Uses both visual and textual information for retrieval and response generation.
* ⏳ **Frame Selection Mechanism:** Efficiently selects key frames from (long) videos.
* ⏳ **Textual Information Extraction:** Leverages textual cues when subtitles are missing.
* ✅ **State-of-the-Art Performance:** Outperforms existing baselines on multiple benchmarks.

## 📖 Abstract

Retrieval-Augmented Generation (RAG) is a powerful strategy for improving the factual accuracy of models by retrieving external knowledge relevant to queries and incorporating it into the generation process. However, existing approaches primarily focus on text, with some recent advancements considering images, and they largely overlook videos, a rich source of multimodal knowledge capable of representing contextual details more effectively than any other modality. While very recent studies explore the use of videos in response generation, they either predefine query-associated videos without retrieval or convert videos into textual descriptions losing multimodal richness. To tackle these, we introduce VideoRAG, a framework that not only dynamically retrieves videos based on their relevance with queries but also utilizes both visual and textual information. The operation of VideoRAG is powered by recent Large Video Language Models (LVLMs), which enable the direct processing of video content to represent it for retrieval and the seamless integration of retrieved videos jointly with queries for response generation. Also, inspired by that the context size of LVLMs may not be sufficient to process all frames in extremely long videos and not all frames are equally important, we introduce a video frame selection mechanism to extract the most informative subset of frames, along with a strategy to extract textual information from videos (as it can aid the understanding of video content) when their subtitles are not available. We experimentally validate the effectiveness of VideoRAG, showcasing that it is superior to relevant baselines.

## 📌 Get Started

We will release the details soon.

## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{jeong2025videorag,
  title={VideoRAG: Retrieval-Augmented Generation over Video Corpus},
  author={Jeong, Soyeong and Kim, Kangsan and Baek, Jinheon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2501.05874},
  year={2025}
}
```
