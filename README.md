# Project: Reasoning Audits with Benchmarks 1

### Team: Amine Turki, Daniel Wei Li, Mingxuan Tang, Raj Dhyaneshvar Chennai Jayamoorthy
### Professors: Edmon Begoli, Gomathi Lakshmanan

## Environment setup

```bash
conda create --name env_hallu python=3.11
conda activate env_hallu
pip install -r requirements.txt
```
## LLM judge prototypes
The code goes first through each prompt in truthfulQA dataset, specifically the multiple choice section. It asks gpt-3.5 to make a short response under 30 tokens. The response is recorded and fed to a LLM-judge which is gpt-40 here, that decides if it thinks gpt-3.5 has hallucinated or not. After that, we compute these text features:
 - Prompt length
 - anwser length
 - Number of question marks
 - Contains date
 - Contains 'who'
 - Contains 'what'
 - Contains 'why'
 - Contains 'unknows'

 We feed these features into random forest and compute feature importance. 

## Dataset
The dataset was computed using the [HalluLens GitHub repository](https://github.com/facebookresearch/HalluLens).  o4-mini was used to generate question-answer pairs using task 1 and task 3 in [HalluLens GitHub](https://github.com/facebookresearch/HalluLens). GPT-5 was utilized to detect hallucination. Task 1 focuses on evaluating the ability of the model to answer precise facts from Wikipedia without hallucinating. Task 3 evaluates the ability to recognize non-existent entities. 

The generated data is in 4 json files (dataset/hallulens_data folder):
 - task1_merged_generation.jsonl: question-answer generated using task 1
 - task1_merged_hallucination_scores.jsonl: gives where hallucination was detected with task 1
 - task3_merged_generation.jsonl: question-answer generated using task 3
 - task3_merged_hallucination_scores.jsonl: gives where hallucination was detected with task 3

### Dataset sample

### Dataset Preview (4000 rows × 3 columns)

| Index | prompt | halu_test_res | cot_steps |
|-------|--------|----------------|-----------|
| 0 | What position did Jan Willem Spruyt hold in the ... | 1 | 1. I recognized that the prompt is asking for ... |
| 1 | What was the new name given to First Prize by ... | 0 | 1. **Identifying the Key Terms**: I recognized... |
| 2 | What is the northern terminus of Mississippi H... | 1 | 1. **Identification of the Highway**: I recogn... |
| 3 | Who is betrothed to St. Elmo in the film? | 1 | 1. **Identify the Subject**: I recognized that... |
| 4 | What date was the album Oddfellows released? | 0 | 1. I recognized that the prompt is asking for ... |
| ... | ... | ... | ... |
| 3995 | I'd like to learn more about the plant Thelypo... | 1 | 1. **Identify the Subject:** The prompt reques... |
| 3996 | Could you provide an overview of the plant Bar... | 1 | 1. **Identify the Plant**: The model recognize... |
| 3997 | I want to know more about the plant Brunfelsia... | 1 | 1. **Identifying the Plant**: I recognized tha... |
| 3998 | What can you tell me about the plant Munroa gl... | 1 | 1. **Identification of the Species**: I recogn... |
| 3999 | Can you describe the plant Microlejeunea augieri? | 1 | 1. **Identifying the Subject**: I recognized t... |

 - 0: No hallucination detected
 - 1: Hallucination detected

## Input text features extraction
File feature_extraction.ipynb consists of a Jupyter notebook where we compute different text features using the following packages:
 - Spacy: textual and syntactic features
 - Textstat: text complexity and readability
 - VaderSentiment: sentiment and emotional analysis
 - Embedding

Since the dataset is small, we decided to use LogisticRegression, and because we are dealing with a binary classification (label 0: no hallucinaiton detected, label 1: hallucination detected) to compute features importance. 

### Text Features computed


| **Category** | **Variable Name** | **Brief Description** |
|--------------|-------------------|------------------------|
| **Basic Counts** | `number sentence` | Number of sentences in the text. |
| | `num_tokens` | Count of tokens excluding spaces. |
| | `char_length` | Total character length of the text. |
| | `num_entities` | Total number of named entities detected. |
| | `num_numbers` | Count of numeric tokens. |
| **Vocabulary & Structure** | `lemma ratio` | Ratio of unique lemmas to total tokens (vocabulary richness). |
| | `avg_sentence_length` | Average number of tokens per sentence. |
| | `max_sentence_length` | Maximum length among all sentences. |
| **POS Ratios** | `noun_ratio` | Proportion of tokens that are NOUN. |
| | `verb_ratio` | Proportion of tokens that are VERB. |
| | `adjective_ratio` | Proportion of tokens that are ADJ. |
| | `adverb_ratio` | Proportion of tokens that are ADV. |
| | `auxiliairis_ratio` | Ratio of auxiliary verbs (AUX). |
| **Token Category Ratios** | `stop_ratio` | Ratio of stopwords in the text. |
| | `punctuation_ratio` | Ratio of punctuation tokens. |
| | `number_ratio` | Ratio of numeric tokens. |
| **Named Entity Ratios** | `entity_ratio` | Ratio of named entities to total tokens. |
| | `person_ratio` | Ratio of "PERSON" entities to total tokens. |
| | `org_ratio` | Ratio of "ORG" entities to total tokens. |
| | `gpe_ratio` | Ratio of "GPE" entities to total tokens. |
| | `date_ratio` | Ratio of "DATE" entities to total tokens. |
| **Readability (textstat)** | `flesch score` | Flesch Reading Ease score. |
| | `grade (US School)` | US school grade readability level. |
| | `complexity` | Gunning Fog complexity index. |
| | `Number rare words` | Count of difficult/rare words. |
| | `Dale-Chall Score` | Dale–Chall readability score. |
| **Sentiment (VADER)** | `negative sentiment` | Sentiment intensity for negative tone. |
| | `neutral sentiment` | Sentiment intensity for neutral tone. |
| | `positive sentiment` | Sentiment intensity for positive tone. |
| | `overall sentiment` | Compound sentiment polarity score. |
| **Embedding-Based Similarity** | `cosine similarity prompt-CoT` | Cosine similarity between prompt embedding and CoT embedding. |
| | `ce_stsb_roberta prompt-cot` | Cross-encoder semantic similarity using STSB-RoBERTa-Large. |
| | `bleurt` | BLEURT semantic similarity score between prompt and CoT. |



### Features importance computation
Feature importance was computed using:
 - Logistic regression 
 - Matrix correlation
 - Mutual information
 - Random Forest
 - XGBoost
 - SVM with RBF Kernel
 - MLP Classifier

## License
Based on Hallulens github license section: "The majority of HalluLens is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/shmsw25/FActScore is licensed under the MIT license; VeriScore is licensed under the Apache 2.0 license."


## Reference
```bibtex
@article{bang2025hallulens,
  title={HalluLens: LLM Hallucination Benchmark},
  author={Yejin Bang and Ziwei Ji and Alan Schelten and Anthony Hartshorn and Tara Fowler and Cheng Zhang and Nicola Cancedda and Pascale Fung},
  year={2025},
  eprint={2504.17550},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.17550},
}

@article{honnibal2017spacy,
  title={spaCy 2: Natural Language Understanding with Bloom Embeddings, Convolutional Neural Networks, and Incremental Parsing},
  author={Matthew Honnibal and Ines Montani},
  year={2017},
  note={To appear},
  url={https://spacy.io/}
}

@software{bansal2025textstat,
  title={textstat: A Python library for computing text readability and complexity metrics},
  author={Shivam Bansal and Chaitanya Aggarwal},
  year={2025},
  version={0.7.10},
  url={https://pypi.org/project/textstat/},
  note={Python Package Index (PyPI)}
}

@software{hutto2014vader,
  title={VADER: A Parsimonious Rule‐based Model for Sentiment Analysis of Social Media Text},
  author={C. J. Hutto and E. E. Gilbert},
  year={2014},
  note={Used via vaderSentiment (version 3.3.2) Python package},
  url={https://pypi.org/project/vaderSentiment/}
}

@article{harris2020array,
  title={Array programming with {NumPy}},
  author={Charles R. Harris and K. Jarrod Millman and Stéfan J. van der Walt and Ralf Gommers and Pauli Virtanen and David Cournapeau and Eric Wieser and Julian Taylor and Sebastian Berg and Nathaniel J. Smith and others},
  journal={Nature},
  volume={585},
  pages={357--362},
  year={2020},
  doi={10.1038/s41586-020-2649-2}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine Learning in Python},
  author={Fabian Pedregosa and Gaël Varoquaux and Alexandre Gramfort and Vincent Michel and Bertrand Thirion and Olivier Grisel and Mathieu Blondel and Peter Prettenhofer and Ron Weiss and Vincent Dubourg and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}

@article{mckinney2010data,
  title={Data Structures for Statistical Computing in Python},
  author={Wes McKinney},
  journal={Proceedings of the 9th Python in Science Conference},
  pages={56--61},
  year={2010},
  editor={Stéfan van der Walt and Jarrod Millman}
}

@software{python,
  title = {Python: A Programming Language for Scientific Computing},
  author = {{Python Software Foundation}},
  year = {2025},
  url = {https://www.python.org/}
}


@misc{openai2023api,
  author = {OpenAI},
  title = {OpenAI API: Python Client Library},
  year = {2023},
  url = {https://github.com/openai/openai-python},
  note = {Accessed via the OpenAI API at https://platform.openai.com}
}

@inproceedings{lhoest2021datasets,
  title = {Datasets: A Community Library for Natural Language Processing},
  author = {Quentin Lhoest and Albert Villanova del Moral and Yacine Jernite and Abhishek Thakur and Patrick von Platen and Julien Chaumond and Mariama Drame and Julien Plu and Lewis Tunstall and Joe Davison and others},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages = {175--184},
  year = {2021},
  url = {https://github.com/huggingface/datasets}
}

@article{breiman2001randomforest,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001},
  publisher={Springer}
}

@inproceedings{reimers2019sentencebert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Nils Reimers and Iryna Gurevych},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019},
  url={https://arxiv.org/abs/1908.10084}
}

@inproceedings{reimers2020crossencoder,
  title={Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation},
  author={Nils Reimers and Iryna Gurevych},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020},
  note={Used for STSB-RoBERTa CrossEncoder},
  url={https://arxiv.org/abs/2004.09813}
}

@article{sellam2020bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  journal={Transactions of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011},
  url={https://scikit-learn.org/}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Tianqi Chen and Carlos Guestrin},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2016},
  url={https://dl.acm.org/doi/10.1145/2939672.2939785}
}

@article{cover1991information,
  title={Elements of Information Theory},
  author={Thomas M. Cover and Joy A. Thomas},
  year={1991},
  note={Standard reference for Mutual Information},
  publisher={Wiley},
  url={https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X}
}

@article{pearson1895correlation,
  title={Note on Regression and Inheritance in the Case of Two Parents},
  author={Karl Pearson},
  journal={Proceedings of the Royal Society of London},
  year={1895},
  note={Classic reference for correlation coefficient},
  url={https://royalsocietypublishing.org/doi/10.1098/rspl.1895.0041}
}
