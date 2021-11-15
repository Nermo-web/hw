# hw
I choose the running, sitting, standing and walking data set. And use neuron network to classify the four classes. The data set is csv file, so we just use os.lisdirt to get all the files in current directory, and use if running is in filename and so on to get the corresponding four files and their labels. In this way, we just get the data set. Now, I just need to preprocess the data. We transform the data to be zero mean and variance 1 by subtracting the mean and dividing the standard deviation. Now the data can be feed to the neuron network. I used four hidden layers, each hidden layer has four neurons, and we just use the relu active function. 
I used the adam algorithm to optimize the parameters. And the result is follow:
 

From above, we know that the model indeed learn something. 


We use 0.01 proportion of the data to be the validation data. This proportion can be changed by modifying the parameter ratio in the code.

Conclusion:
I found that although the value of the raw data is very big, but if we do preprocess to scale it in to some learnable range, the result will be very good, but I do also find that if I do not preprocess the raw data, the result can be poor, it is because the parameter can be easily affected by noise.

 
Dataset resources used:

https://sensor.informatik.uni-mannheim.de/#dataset_realworld_subject2

https://sensor.informatik.uni-mannheim.de/#dataset_realworld
