# Covid-19-Question-Answering
Covid-19 Question Answering using TAPAS
Developing Longformer based question answering model.
Trying to overcome the shortcomings caused by BERT’s input token limit(512).
The goal is to beat TAPAS and TABERT models on large tabular data.

# Product-Matching





<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#data-description">Data Description</a></li>
    <li><a href="#files">Files</a></li>
    <li><a href="#Pre-Processing">Pre-Processing</a></li>
    <li><a href="#Results">Results</a></li>
    <li><a href="#future-work">Future Work</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Tabular data is difficult to analyze and search through, models are typically trained on free-form
NL text, hence may not be suitable for tasks like semantic parsing over structured data, which
require reasoning over both free-form NL questions and structured tabular data (e.g., database
tables). Traditionally, answering language questions over tables is done by using full logic. But
recently advanced models like TAPAS and TABERT use a better way to solve the problem. In
this work we use TAPAS as the model for the question answering task on COVID-19 dataset.
The dataset is a collaborative volunteer-run effort to track the ongoing COVID-19 pandemic by
The Atlantic. After the experiments we conclude that the TAPAS model gives accurate results on
reasoning type questions and a fine-tuned version of the same gives satisfactory results on
retrieval type questions. The question set is self prepared and has 25 questions with correct
answers marked along with corresponding cells for fine tuning the model.

### Built With

* Python
* Pytorch
* Hugging Face



<!-- GETTING STARTED -->
## Introduction

Since the COVID-19 outbreak, there has been lack of proper information. For example if we google search “Which state has the minimum number of ICU patients in the US” we are not provided with a straightforward answer, rather we are given a set of articles. The sources for articles vary from one another which certainly makes it more difficult to find the accurate information. To alleviate this problem and provide accurate information we propose a fine-tuned model on COVID dataset to provide accurate answers for the corresponding questions. 
Question answering on tabular data is a difficult task on which many researchers are working. Traditionally, the approach is to convert the question to a structured query and retrieve the answer. But this approach has a fundamental problem that the question has to be grammatically correct and the query must be syntactically accurate, even if there is a slight change in the question the relevant query might not be produced. Due to rapid advancement in natural language processing with the introduction of transformers, the new way of generating answers from a given question has developed. In this work I have used a TAPAS, a weakly supervised table parsing model which is developed based on BERT. TAPAS has proved effective
on sequential question answering tasks and performs comparatively well on table retrieval questions with respect to other state-of-the-art architectures.

### Prerequisites

Kaggle Shopee - Price Match Guarantee Dataset

## Related Work
Traditionally, for natural language processing, we have used the sequence to sequence models which depend on spatial dependencies and will process inputs sequentially. Which has a major drawback of not being able to parallelize tasks and also having vanishing gradients problems. Many architectures like LSTM were developed, but they fundamentally use the same concept. In 2017 the new novel architecture (Transformer) has been proposed by Ashish Vaswani, which relies entirely on self-attention and process input all at once. 

## TAPAS 
TAPAS is an extension of BERT architecture with two additional classification layers which helps it predict the aggregation operator and the cells on which they have to be used. TAPAS encodes the question along with a table as input, also, it uses additional token embeddings ( Segment, Column, Row, and Rank) which helps encode the table structure.

The model predicts the aggregation operator along with the selection. The aggregation operator (NONE, SUM, COUNT, AVERAGE) describes the operation to be performed on the selected cells. 

### Approach for fine-tuning 
There are three approaches for fine-tuning the model based on the task in which we want the model to work. 
| Task   | Example Dataset Description | 
| ------------- | ------------- |
| Conversational   | SQA Conversational, only cell selection questions  |
| Weak supervision for aggregation | WTQ Questions might involve aggregation, and the model must learn this given only the answer as supervision| 
| Strong Supervision for aggregation | WikiSQL-supervised Questions might involve aggregation, and the model must learn this given the gold aggregation operator|

 
### Data Description

I have used four different datasets provided by the COVID tracking project by The Atlantic. The main dataset consisted of 21k rows and 15 columns of data collected till March 7th, 2021 across all 56 states in the US.  


### Approach
My objective is to do a detailed analysis of the TAPAS model on the COVID dataset. I have first pre-processed the data and selected useful features(the datasets had columns that are not significant, I have excluded those columns). 
For fine-tuning the model, the dataset must be in SQA format having columns ( id, annotator, position, question, table_file, answer_coordinates, answer_text, aggregation_label(Optional), float_answer) 
I have prepared 25 questions along with answer coordinates corresponding to the table used. The questions were broadly divided into two categories 1) Retrieval 2) Reasoning

### Retrieval Questions
These types of questions involve selecting respective columns and performing the predicted aggregation operation on them. For example “Deaths in the US on 2021-03-07” for which the model will predict the corresponding cells and also an aggregation operator. 

### Reasoning Questions 

These types of questions involve only cell selection for the corresponding table. For example “Which state has minimum number of patients in ICU?” for which the model will give the corresponding cell selection and NONE as aggregation operator. 
I have used two models 1) Not fine-tuned 2) fine-tuned on COVID dataset 
#### Not fine-tuned on COVID dataset(Non-FT):
I used the pretrained tapas base model which is fine-tuned on SQA, WikiSQL, and finally WTQ datasets. The large model and base model have almost similar accuracies (.5062 and .4525), so I used the base model for inference. 
The fine-tuned model did not perform well as compared to the default fine-tuned model on WTQ. The main reason being we did not have enough data to effectively fine-tuneNon-FT the pre-trained model. 


#### Fine-tuned on COVID dataset (FT) : 
At first, I used the TAPAS base model which is pretrained on a large Wikipedia English corpus, and fine-tuned it on the COVID dataset. I have prepared a set of questions and along with that the cell coordinates of the correct answer and fine-tuned it on with questions and corresponding dataset.

### Improvement
Instead of fine-tuning the pretrained model for fine-tuning on the COVID dataset, I have used the fine-tuned model on WTQ. I have used a very low learning rate of 0.0000193581 as it is already fine-tuned on a large corpus of wiki table questions. This fine-tuning showed significant results as compared to the non fine-tuned model. 

## Hyperparameters
Number of aggregation labels- 4 (none, sum, average, count) 
Use answer as supervision- True (Whether to choose ground truth as supervision for aggregation examples) 
Answer loss cutoff- None (Default “0.664694”: Ignore examples with answer loss larger than cutoff) 
Cell selection preference- 0.207951 (default- outputs an aggregator operator if the probability is greater than the given number else it is NONE) 
Learning rate- 0.0000193581 (default - 5e-5) 
Optimizer- Adam

<!-- USAGE EXAMPLES -->
## Conclusion 

In this work, I have used TAPAS for question answering on tabular COVID data. Results show that for reasoning questions the model prediction is accurate for both fine-tuned and non fine-tuned versions. For retrieval questions, the fine-tuned model predicted accurate answers where it had a clear indication of which cell to retrieve. Also in general questions, the accuracy of the answer depends on the clarity of the question, columns in the dataset, and on which data the model is fine-tuned on. 
Furthermore, the major drawback of this model is the input( question+ table columns+ table rows) should not exceed 512 tokens which are not suitable for many practical purposes. So in future work, we try to alleviate this problem by using different architectures. 


<!-- ROADMAP -->
## Future Work

TAPAS uses a BERT architecture that suffers from an input token limit of 512. The tokenized input separator tokens table columns+table cells all must be in the limit of 512, which is not feasible for practical use. To alleviate this problem, we can use Longformers which uses an attention mechanism that scales linearly with sequence length. It uses local windowed attention with task-motivated global attention and it has an input token limit of 4096. 






<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Shiva Maruth Alapati - alapati.shiva1998@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Kaggle](https://www.kaggle.com/)
* [Hugging Face](https://huggingface.co/)


