# Calgacus
Implementation of the steganographic protocol presented in the paper [*LLMs can hide text in other text of the same length*](https://arxiv.org/abs/2510.20075).

## Instructions

Click on the big blue button [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noranta4/calgacus/blob/main/LLMs_can_hide_text_in_other_text_of_the_same_length.ipynb) and run all the cells of the notebook. That's it.

It is a self-contained notebook useful to run the Calgacus protocol using different LLMs. The free GPU runtime of colab is sufficient to run good LLMs fully on GPU and encode/decode paragraphs in seconds, even from smartphone. To fully reproduce the stegotexts of the paper, run the notebook on a NVidia Ada GPU (e.g. RTX40XX) using ```llama-cpp-python==0.3.12```.

## Paper

PDF: [*LLMs can hide text in other text of the same length*](https://arxiv.org/pdf/2510.20075)

By: [Antonio Norelli](https://noranta4.com/) and [Michael Bronstein](https://www.cs.ox.ac.uk/people/michael.bronstein/)

**Abstract**: A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.

<img src="https://github.com/noranta4/calgacus/blob/main/method.png" alt="Image" width="800">


## Cite
If you liked our work and want to cite it in yours:
```
@article{norelli2025llms,
  title   = {LLMs can hide text in other text of the same length},
  author  = {Antonio Norelli and Michael Bronstein},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2510.20075}
}
```

