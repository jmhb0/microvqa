# MicroVQA
This is the code for the MicroVQA benchmark (hosted on [🤗HuggingFace here](https://huggingface.co/datasets/jmhb/microvqa)) and the 🤖RefineBot method that removes language shortcuts from multiple-choice evaluations. They were published in the paper: [MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research](https://jmhb0.github.io/microvqa/). 

The repo contains:
- `eval` evaluation code for the MicroVQA benchmark (hosted on [🤗HuggingFace here](https://huggingface.co/datasets/jmhb/microvqa)). See its [README](eval/README.md).
- `refinebot` is the 🤖RefineBot method for removing language shortcuts from MCQs. See its [README](refinebot/README.md).
- `benchmark` is code used in benchmark construction. See its [README](benchmark/README.md)

If any of this is useful, please cite us!
```
@article{@burgess2025microvqa,
      title={MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research}, 
      author={James Burgess and Jeffrey J Nirschl and Laura Bravo-Sánchez and Alejandro Lozano and Sanket Rajan Gupte and Jesus G. Galaz-Montoya and Yuhui Zhang and Yuchang Su and Disha Bhowmik and Zachary Coman and Sarina M. Hasan and Alexandra Johannesson and William D. Leineweber and Malvika G Nair and Ridhi Yarlagadda and Connor Zuraski and Wah Chiu and Sarah Cohen and Jan N. Hansen and Manuel D Leonetti and Chad Liu and Emma Lundberg and Serena Yeung-Levy},
      year={2025},
      eprint={2503.13399},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.13399}, 
}
```
