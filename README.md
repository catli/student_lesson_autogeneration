<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js? 
config=TeX-MML-AM_CHTML"
</script>

# Student Lesson Path-Finder
[*Insight AI Fellowship Project*] The purpose of this project is to auto-generate lessons for students using Khan Academy. Every week, millions of students access Khan Academy for its free educational content. However because there are thousands of different exercises, videos and articles, students can easily lose track of how to progress through the content. The goal of this project is to try to learn optimal paths from highly motivated students. To prediction uses Gate Recurrent Unit (GRU) model, a type of RNN, to predict optimal future session activities.



### How It Works
The goal of the model is to learn how highly-motivated students progress through Khan Academy, and use that information to curate a set of lesson for future students navigating the site.

The model uses a type of Recurrent Neural Network (RNN) called Gated Recurrent Unit (GRU) to learn the representation of student pathways. Because RNN models are able to hold memory over many prediction steps, they are often used for sequence prediction, such as words in a sentence or levels in a game. The PathFinder customizes this model so it can be used to predict student learning sequences.



#### GRU Model
I selected the GRU model, bedcause it can transfer memory from previous input steps to future steps. Below is a diagram representing how the GRU cell at each step works. At time step $t$, the model input the input $$x_t$$, a hidden layer from the previous $$h_(t-1)$$. The hidden layer represents the memory passed from previous sessions. The update gate $$z_t$$ is then used to decide what to keep and what to throw out.

![alt text](png/gru_colah.png "source: Chris Colah's blog post")

The PathFinder adiopts the GRU model to predict lessons on Khan Academy for students. Below is a diagram of how the PathFinder predicts activities for a student's third session. Each session is represented as a vector or embedding describing all the activities they worked on and what percent of questions they answered correctly. At each time step, the GRU model reads the vector representing the last session, along with a hidden layer representing the stored memory of previous sessions. The model generates an output and a new hidden layer based on on what it wants to throw out and keep. By having the flexibility to store different components of previous activities in memory, the GRU model generates much better than predicting on just the last session alone.

![alt text](png/pathfinder_gru.png "How PathFinder Works")

For each output and predicted session activities, the model validates recommended activities against actual activities the learner selected. Since the model is trained and validated on highly-motivated students, the assumption is that if a model is able to predict their selected activities with high recall and precision, it is picking the optimal pathway. Because student pathways can be quite noisy and the potential selected activities quite high, the precision and recall will not be at the levels we expect in language models or other use cases.



### Data Description

The model is trained on anonymized dataset Khan Academy student sessions from March to June 2018. This data is not publicly avaialble, but a dummy play dataset is available in this repo.

Input: The model expects a series of tokens representing activities for each session and an index that will allow it to translate
the tokens back to the activity names.

For example, Nadia might have 3 sessions that looks like the following:
```
    Session 1: equivalent fractions (50%), recognizing fractions (75%)
    Session 2: equivalent fractions (100%), recognizing fractions (100%)
    Session 3: equivalent fractions (95%), recognizing fractions (100%), cutting shapes into parts (75%)
```

And the activities map to the following tokens in the index:

```
    { recognizing fractions: 1, equivalent fractions: 2, cutting shapes into parts: 4}
```

Based on the token mapping and the session activities, the input for Nadia will look like this:

```
    {'Nadia': {'Session 1': [(2, 0.5), (1, 1.0)],
               'Session 2': [(2, 1.0), (1, 1.0)],
               'Session 3': [(2, 0.95), (1, 1.0), (3, 0.75)]}
```




### What are the files in this repo

`input`: Directory with files storing model parameters
> `my_env.yml`: Stores conda environment
> `model_params.yaml`: Stores model parameters for `train.py`
> `predict_params.yaml`: Stores prediction parameters for `predict.py`

`model`: Directory that runs the GRU model on tokenized data
> `train.py`: Run training model and evaluate the test result
> `gru.py`: Define the GRU model class, with forward propagation and loss function
> `evaluate.py`: Function to evaluate loss
> `process_data.py`: Function to convert ingested token data into model input vector
> `predict.py`: Run inference on test data using a trained model

> `logisticnn.py`: Run a logistic model prediction, not used for GRU model but provides baseline performance

`data`: Directory to process raw data into tokenize form, with available index
`summary`: Summarize dataset and perform kmeans-cluster on raw student data 



### How to run model and inference

#### Set-up
1. Clone repo into your local machine

2. Create a conda environment with the appropriate specifications to run file
    ` conda env create -f environment.yml`

#### Training
[TODO] add training

#### Inference
[TODO] add inference steps


### Results




### Resources
The model is inspired by a couple of different pieces of work and the resources are available here:

[RNN for game generation](https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3)  
[Deep Knowledge Tracing](https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)  
[GRU Model](https://arxiv.org/pdf/1406.1078.pdf)



### Workstream

Below is the snapshot of how this project was built up and organized.

![alt text](png/asana_wk1_2.png)

![alt text](png/asana_wk3_4.png)

