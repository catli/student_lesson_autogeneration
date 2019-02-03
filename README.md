# Student Lesson Autogeneration
[*Insight AI Fellowship Project*] The purpose of this project is to explore approaches to auto-generate lessons for students learning on Khan Academy, a nonprofit educational website with thousands of free content. There are currently millions of individuals using Khan Academy, but because of the large variety of content and subjects available, students can easily lose track of how to progress through the product. The goal of this project is to try to learn optimal paths from prevous highly motivated students. To prediction uses Gate Recurrent Unit (GRU) model, a type of RNN, to predict optimal future session activities.  


### What is the Model Objective
The model built will learn how highly-motivated learners progress through Khan Academy on their own, and use that information to curate a set of lesson for active learners.  

For each learner, the model tries to input lessons prior to the target lesson, and predict relevant activities for future lessons. To validate on the outcome, the model compares the prediction against actual choose activities for highly motivated learners. 

The GRU model works by combining information from the students' last session with a hidden memory layer representing a selected representation of earlier sessions. By having the flexibility to incorporate memory of previous activities, the GRU model generates much better prediction than a model just looking at the last session. 

Here's the layout of the model 

[TODO: Add visual]





### Data Description
Input: The model expects a series of one-hot vectors or embeddings. Each vector describes a single student session, include what activity they spent time on and, if they work on an exercise what percent of the 
Data contains x learners over 3 months, and y subjects. 

Output: The model outputs an embedding for each subsequent session after the first session ... 
  [TODO: Add example output]


### What are the files in this repo

> `model_params.yaml`: Stores the model parameters used to run `train.py`   

> `my_env.yml`: Stores the conda environment specifications

`model`: Run the GRU model on tokenize data 
> `train.py`: Run training model and evaluate the test result
> `logisticnn.py`: Run a logistic model predicting next session from previous session, sets baseline for GRU model performance

`data`: Process raw data into tokenize form, with available index
  
`summary`: Summarize dataset and perform kmeans-cluster on raw student data 

### How to test and run model



#### Resources
The model is inspired by a couple of different pieces of work and the resources are available here:

[RNN for game generation](https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3)  
[Deep Knowledge Tracing](https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)  
[GRU Model](https://arxiv.org/pdf/1406.1078.pdf)



### Workstream
[TODO: Add asana image of workstream]
