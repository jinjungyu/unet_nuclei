# CH1. What is Deep Learning?
## 1.1 Artificial intelligence, Machine learning, and Deep learning
### 1.1.1 Artificial Interlligence
- AI is a general field that encompasses machine learning and deep learning, but that also includes many more approaches that don’t involve any learning.
- 
- symbolic AI: For a fairly long time, many
experts believed that human-level artificial intelligence could be achieved by having programmers handcraft a sufficiently large set of explicit rules for manipulating knowledge. it was the dominant paradigm in AI from the 1950s to the late 1980s. it turned out to be intractable to figure out explicit rules for solving more complex problem.
### 1.1.2 Machine Learning
- In classical programming, the paradigm of symbolic AI, humans input rules (a program) and data to be processed according to these rules, and out come answers. With
machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce original answers.
- A machine-learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and it finds statistical structure in these examples that eventually allows the system to come up with rules for automating the task.
- machine learning, and especially deep learning, exhibits comparatively little mathematical theory-maybe too little—and is engineering oriented.
### 1.1.3 Learning representations from data
the difference between deep learning and other machine-learning approaches
> In Machine Learning
> 1. *Inputs data points* - If the task is image tagging,
they could be pictures.
> 2. *Examples of the expected output* - In an image task, expected outputs could be tags such as “dog,” “cat,” and so on.
> 3. *A way to measure whether the algorithm is doing a goot job* - This is necessary in order to determine the distance between the algorithm’s current output and its expected output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning
-  the central problem in machine learning and deep learning is to meaningfully transform data. in other words, to learn useful representations of the input data at hand - representations that get us closer to the expected output.
#### Example
Consider an x-axis, a y-axis, and some points represented by their coordinates in the (x, y) system.
- The inputs are the coordinates of our points.
- The expected outputs are the colors of our points
- A way to measure whether our algorithm is doing a good job could be, for instance, the percentage of points that are being correctly classified.
if we used as feedback the percentage of points being correctly classified, then we would be doing machine learning. in the context of machine learning, Learning describes an automatic search process for better representations.
- Machine-learning algorithms aren’t usually creative in finding these transformations; they’re merely searching through a predefined set of operations, called a hypothesis space. So that’s what machine learning is, technically: searching for useful representations of some input data, within a predefined space of possibilities, using guidance
from a feedback signal.
### 1.1.4 The "deep" in deep learning
Deep learning is a specific subfield of machine learning and puts an emphasis on learning successive layers of increasingly meaningful representations.
- The *deep* in deep learning stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the *depth* of the model.
- In deep learning, these layered representations are (almost always) learned via models called neural networks, structured in literal layers stacked on top of each other.
- So that’s what deep learning is, technically: a multistage way to learn data representations. It’s a simple idea—but, as it turns out, very simple mechanisms, sufficiently
scaled, can end up looking like magic. 
### 1.1.5 Understanding how deep learning works, in three figures
The specification of what a layer does to its input data is stored in the layer’s *weights*. we’d say that the transformation implemented by a layer is parameterized by its weights. In this context, *learning* means finding a set of values for the weights of all layers in a network  
a deep neural network can contain tens of millions of parameters. 
- you need to be able to measure how far this output is from what you expected. This is the job of the *loss function* of the network. The *loss function* takes the predictions of the network and the true target and computes a distance score
- The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the *loss score* for the current example. This adjustment is the job of the *optimizer*, which implements what’s called the *Backpropagation* algorithm
- Initially, the weights of the network are assigned random values. every example the network processes, the weights are adjusted a little in the correct direction, and the loss score decreases. This is the *training loop*, which, repeated a sufficient number of times, yields weight values that minimize the loss function.

## 1.2 Before deep learning: a brief history of machine learning
It’s safe to say that most of the machine-learning algorithms used in the industry today aren’t deep-learning algorithms. Deep learning isn’t always the right tool for the job — sometimes there isn’t enough data for deep learning to be applicable, and sometimes the problem is better solved by a different algorithm.
we’ll briefly go over them and describe the historical context in which they were developed. This will allow us to place deep learning in the broadercontext of machine learning and better understand where deep learning comes from and why it matters.
### 1.2.1 Probabilistic modeling
Probabilistic modeling is the application of the principles of statistics to data analysis. ex) Naive Bayes algorithm.  
