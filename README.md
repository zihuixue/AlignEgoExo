# AlignEgoExo (AE2)
[**Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment**](https://vision.cs.utexas.edu/projects/AlignEgoExo/)                                     
Zihui Xue, Kristen Grauman      
NeurIPS, 2023  
[project page](https://vision.cs.utexas.edu/projects/AlignEgoExo/) | [arxiv](https://arxiv.org/abs/2306.05526) | [bibtex](#citation)

## Overview
We present AE2, a self-supervised embedding approach to learn fine-grained action representations that are invariant to the ego-exo viewpoints.
<p align="center">
  <img src="images/framework.png" width=80%>
</p>

We propose a new [ego-exo benchmark](#ego-exo-benchmark) for fine-grained action understanding, which consist of four action-specific datasets. For evaluation, we annotate these datasets with dense per-frame labels.
<p align="center">
  <img src="images/dataset.png" width=80%>
</p>

## Ego-Exo Benchmark
We assemble ego and exo videos from five public datasets and collect a ego tennis dataset. Our benchmark consists of four action-specific ego-exo datasets:
+ (A) Break Eggs: ego and exo videos from [CMU-MMAC](http://kitchen.cs.cmu.edu).
+ (B) Pour Milk: ego and exo videos from [H2O](https://taeinkwon.com/projects/h2o/).
+ (C) Pour Liquid: ego pour water videos from [EPIC-Kitchens](https://epic-kitchens.github.io/2023) and exo pour videos from [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview). 
+ (D) Tennis Forehand: ego videos we collect and exo tennis forehand videos from [Penn Action](http://dreamdragon.github.io/PennAction/).


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{xue2023learning,
      title={Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment},
      author={Xue, Zihui and Grauman, Kristen},
      journal={NeurIPS},
      year={2023}
}
```