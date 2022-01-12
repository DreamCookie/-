#!/usr/bin/env python
# coding: utf-8


import numpy
import scipy.special # библиотека scipy.special содержит сигмоиду expit() 




# neural network class definition
class neuralNetwork:
    
    
    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задание узлов в каждом из слоев (входной выходной и скрытый)
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # связать матрицы 
        # вес от узла к узлу
        # w11 w21
        # w12 w22 и тд
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # тренировка
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # запрос нейронной сети
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



# число узлов
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)



data_file = open("mnist_test.csv", 'r')
data_list = data_file.readlines()
data_file.close()


len(data_list)
data_list[0]

import numpy
import matplotlib.pyplot
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')





all_values = data_list[0].split(',') 
image_array = numpy.asfarray(all_values[1:]).reshape((28,28)) 
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')






scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)





# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# коэффициент обучения равен 0,3
learning_rate =0.3
# создать экземпляр нейронной сети
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
# загрузить в список тестовый набор данных CSV-файла набора MNIST
training_data_file = open("mnist_test.csv", 'r')
training_data_list = training_data_file.readlines ()
training_data_file.close()
# тренировка нейронной сети
# перебрать все записи в тренировочном наборе данных
for record in training_data_list:
# получить список значений, используя символы запятой (1,1)
# в качестве разделителей
    all_values = record.split(',')
# масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# создать целевые выходные значения (все равны 0,01, за исключением
# желаемого маркерного значения, равного 0,99)
    targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] - целевое маркерное значение для данной записи
    targets[int(all_values[0])] =0.99
    n.train(inputs, targets)
    pass

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.closed



n.query(numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01


# тестирование нейронной сети
# журнал оценок работы сети, первоначально пустой
scorecard = []
# перебрать все записи в тестовом наборе данных
for record in test_data_list:
# получить список значений из записи, используя символы
# запятой (*,1) в качестве разделителей
    all_values = record.split(',')
# правильный ответ - первое значение
    correct_label = int(all_values[0])
  
# масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# опрос сети
    outputs = n.query(inputs)
# индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
  
# присоединить оценку ответа сети к концу списка
    if (label == correct_label) :
# в случае правильного ответа сети присоединить
# к списку значение 1
        scorecard.append(1)
    else:
# в случае неправильного ответа сети присоединить
# к списку значение 0
        scorecard.append(0)
    pass

