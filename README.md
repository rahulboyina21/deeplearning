Inclass Assignment 01:

'''
Questions to Answer:
1.	What patterns do you observe in the training and validation accuracy curves?
Answer: 
Training accuracy typically increases steadily, while validation accuracy may 
fluctuate. If both rise together, the model is learning effectively, but if 
training accuracy is much higher than validation accuracy, it suggests overfitting, 
meaning the model memorizes the training data but struggles with unseen data.

2.	How can you use TensorBoard to detect overfitting?
Answer: Overfitting can be identified by observing training and validation loss curves if 
validation loss starts increasing while training loss keeps decreasing, the model is 
overfitting. Additionally, a large gap between training and validation accuracy indicates 
poor generalization. TensorBoard’s histograms and weight distributions also help in 
diagnosing overfitting.

3.	What happens when you increase the number of epochs?
Answer: Initially, increasing epochs improves model performance, but after a certain point, 
overfitting occurs causing validation accuracy to stagnate or decrease. Training loss may 
continue decreasing but the model fails to generalize well. 
Early stopping can prevent 
unnecessary training and help maintain optimal performance.

'''

Inclass Assignment 02:

Question 1: Cloud Computing for Deep Learning (20 points)
Cloud computing offers significant advantages for deep learning applications.
(a) Define elasticity and scalability in the context of cloud computing for deep learning. (10 points)
(b) Compare AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio in terms of their deep learning capabilities. (10 points)
Answer: 

	•	Elasticity: It refers to the ability of cloud resources to expand or shrink automatically based on workload demands. For example, if a deep learning model requires more GPUs during training, the cloud can allocate them dynamically and release them after training is done.
	•	Scalability: It means increasing the cloud infrastructure’s capacity to handle growing workloads efficiently. For instance, if a company wants to train larger models, they can add more servers or GPUs to improve performance.
	•	Key Difference: Elasticity is short-term and automatic, while scalability is long-term and planned for future growth.

Feature	   AWS 	                          Google	               Azaure
Ease of use 	Moderate	              High	                   High
AutoML	    Yes	                          Yes	                   Yes
GPU/TPU	Supports GPUs, Inferentia	Supports GPU/ TPU	Supports GPU/ FPGAs
Best For	Large Scale ML Training	Auto ML, Tensor Flow 	Enterpise AI

Summary:
	AWS SageMaker: Best for flexible, large-scale machine learning.
	Google Vertex AI: Best for AutoML and TensorFlow-based models.
	Azure ML Studio: Best for enterprise AI with Microsoft tools.
