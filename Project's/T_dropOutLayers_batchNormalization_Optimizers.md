# Understanding_How_Things_WORK_LOLLLLLLL

##Dropout_Layers
- Consider them like sudden muscle memory kicks in when there is a sudden action takes place like falling of your beloved Mobile phone lol.
- So in the vast land of the ML and DL -> we use this drop out to test our model to fill out the gaps by it's own rather than relaying on the whole set of training data it's more like when your father has let go your bicycle is a normal way to test your skills.
- leeting go of your bicycle in a unknown edge case as what would you do when you have a pot hole is completely new scenario on how to handle the situation with the knowledge you gained so far bicycle riding.
- Same goes for the neural networks as why would i train it whole of the data 100% all the time or atleast once then what's the point of it's learning ,something it should understand and fill the gaps by it's own like if i say 1+1 = 2 then what is 1+1+1 ? you should have an answer or atleast you should know how it is done when the understaing is accurately clear in atomic level then it wouldn't matter the level of complexity it holds you can still figure out the answer.
- So considering this sometimes -> some neurons dropout by leaving some data out and not allowing the data to be passed from few neurons thus inactivating them.


##Batch_Normalization

- Imagine there is class full of toddlers and you have been asked to give me what was your favourite thing and get ready for the wide array of answers that you wouldn't even expect like elephant, antman , lion , insect's to space station which could be too complex for our mind to get around isn't it so we should set the range and domain and catagory to easy analysis like if i ask what is your fav choclate that is far more easy to filter and traverse isn't it,
Same goes for the Batch Normalization bring every data unit into a common ground for better processing of it as the values are in a balanced domain.

##Optimizers
as the name suggest they are needed to optimize few paramters or the data in a specifc way there are few famous optimizers 

1. SGD(Stocastic Gradient Descent): It's good but slow and steady.

2. SGD with momentum : it starts slows but picks up fast.

3. ADAM : Smart one i can say it will adjust dynamically based on how we are progressing.

Alright genius let me ask you a simple question.
IF ADAM is best will we be using it all the time ?
No ADAM is best it's dynamic nature is it's both the pro and con but why?
- ADAM requires more memory when storing every parameter and it's rate of change and every minute details for so detailed analysis could be overwhelmingly high like you can search for gold particles in a handfull of sand not in a desert of sand.
- When it switches patterns dynamically all the time then on a large data set it might not be able to come with a accurate conclusion as it might change pattern very often in this scenario we often use the SGD with Momentum for better result.
- ADAM is too fast what does that mean it will overfit often sometimes it would fail to find a optimal general template or solution.


# Optimizer Cheat Sheet: Choosing the Right One  

## Overview  
Selecting the correct optimizer is important for improving the performance of a machine learning model. The choice depends on factors such as dataset size, model complexity, and training speed.  

---

## Quick Comparison Table  

| Optimizer | Best For | Advantages | Disadvantages | Analogy |
|-----------|---------|------------|---------------|---------|
| **SGD (Stochastic Gradient Descent)** | Large datasets, simple models (Linear Regression, Logistic Regression) | Good generalization, stable training | Converges slowly, may get stuck in local minima | Similar to jogging at a steady pace every day, which takes time but is effective |
| **SGD + Momentum** | Deep learning models (CNNs, NLP models), situations where SGD is too slow | Faster convergence than standard SGD, avoids local minima | Requires tuning of hyperparameters | Similar to rolling a snowball downhill, starting slowly but increasing in speed |
| **Adam (Adaptive Moment Estimation)** | Small to medium datasets, noisy data (Reinforcement Learning, prototyping) | Fast convergence, works with minimal tuning | May overfit, higher memory usage | Similar to a personal trainer who adjusts the exercise routine dynamically |
| **RMSprop** | Recurrent Neural Networks (RNNs), memory-efficient tasks | Good for recurrent models, adaptive learning rates | Less generalizable compared to Adam | Similar to a simple but efficient workout plan that focuses on specific areas |

---

## When to Use Each Optimizer  

### 1. SGD (Stochastic Gradient Descent)  
**Best for:**  
- Large datasets  
- Simple models such as Linear Regression and Logistic Regression  
- Training where strong generalization is required  

**Avoid when:**  
- Fast convergence is needed  
- The loss function has many local minima, as SGD may get stuck  

**Analogy:** Similar to jogging at a steady pace every day, which takes time but is effective.  

---

### 2. SGD with Momentum  
**Best for:**  
- Deep learning models, such as CNNs and NLP applications  
- Situations where standard SGD is too slow  
- Tasks where avoiding local minima is important  

**Avoid when:**  
- Fine-tuning hyperparameters is difficult  
- Very fast training is required  

**Analogy:** Similar to rolling a snowball downhill, which starts slowly but gains speed over time.  

---

### 3. Adam (Adaptive Moment Estimation)  
**Best for:**  
- Small to medium datasets  
- Training with noisy data, such as Reinforcement Learning  
- Situations where quick convergence is necessary  
- Cases where manual learning rate tuning is not preferred  

**Avoid when:**  
- Strong generalization is required, as Adam may overfit  
- Large-scale image recognition tasks (SGD with Momentum is often better)  
- Memory usage is a concern  

**Analogy:** Similar to a personal trainer who adjusts the workout dynamically for fast improvement.  

---

### 4. RMSprop  
**Best for:**  
- Recurrent Neural Networks (RNNs)  
- Memory-efficient tasks  
- Scenarios requiring adaptive learning rates  

**Avoid when:**  
- Generalization is required for multiple tasks (Adam may be more versatile)  

**Analogy:** Similar to a simple but efficient workout plan that focuses on specific areas.  

---

## Summary  
- **Use SGD** for large datasets, simple models, and strong generalization.  
- **Use SGD with Momentum** for deep models where SGD is too slow.  
- **Use Adam** for fast training, noisy data, and minimal hyperparameter tuning.  
- **Use RMSprop** for RNNs and tasks requiring adaptive learning rates.  

---

## Additional Resources  
For more information, refer to:  
- [Andrej Karpathy’s Deep Learning Lectures](https://www.youtube.com/c/AndrejKarpathy)  
- [Google TensorFlow Optimizer Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)  
- [PyTorch Optimizer Guide](https://pytorch.org/docs/stable/optim.html)


##Transfer Learning in Image Recognition
-> In DeepLearning transfer learning is like taking a well trainned model and fine tune it with a specific task of any kind.
