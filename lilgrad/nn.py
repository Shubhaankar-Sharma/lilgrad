from typing import List
import numpy as np
# class Neuron:
#     def __init__(self, n_inputs):
#         self.weights = np.random.randn(n_inputs)
#         self.bias = np.random.randn()
#         self.output = 0

class ActivationFunction:
    def __init__(self, function, function_derivative):
        self.function = function
        self.function_derivative = function_derivative
    
    def forward(self, inputs):
        return self.function(inputs)

    def derivative(self, inputs):
        return self.function_derivative(inputs)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax_derivative(x):
    return np.ones(x.shape)

def softmax(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    exps = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return exps / np.sum(exps, axis=1)[:, np.newaxis]

relu_activation = ActivationFunction(relu, relu_derivative)
softmax_activation = ActivationFunction(softmax, softmax_derivative)

class NeuronLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros(n_neurons)
        self.outputs = np.zeros(n_neurons)

    # returns matrix [[ouput for input 1 ...], [output for input 2 ...], ...]
    # size of inputs x size of neurons
    def forward(self, inputs: np.array):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights.transpose()) + self.biases
        return self.outputs

    def update_weights(self, learning_rate, gradient):
        self.weights -= learning_rate * gradient

    def update_biases(self, learning_rate, gradient):
        self.biases -= learning_rate * gradient

class NeuralNetwork:
    def __init__(self, n_inputs, n_outputs, n_hidden_layers, neurons_per_layer):
        self.layers: List[NeuronLayer] = []
        init_layer = NeuronLayer(n_inputs, neurons_per_layer[0])
        self.layers.append(init_layer)
        for i in range(1, n_hidden_layers+1):
            self.layers.append(NeuronLayer(neurons_per_layer[i-1], neurons_per_layer[i]))
        # final layer
        self.final_layer = NeuronLayer(neurons_per_layer[-1], n_outputs)
        self.activation_function = relu_activation
        self.final_activation_function = softmax_activation

    # o - y
    # prev gradient [[output1 - target1, output2 - target2, ...], [output1 - target1, output2 - target2, ...], ...] avg over batch
    # prev_layer.
    def calc_gradient_layer(self, prev_gradient, layer: NeuronLayer, prev_layer: NeuronLayer):
        current_gradient = np.dot(prev_gradient, prev_layer.weights)
        if len(layer.outputs) != len(current_gradient):
            raise ValueError("Invalid gradient size")
        current_gradient *= self.activation_function.derivative(layer.outputs)
        return current_gradient

    def calc_gradient_weights(self, curr_gradient, layer: NeuronLayer):
        # [[gradient1, gradient2, ... for n neurons], [gradient1, gradient2, ...], ...]^T n x m
        # [[input1, input2, ... for k inputs and k weights per neuron], [input1, input2, ...], ...] m x k
        # we need [[[-weight1, -weight2... k],... ] k x n, ... m batches]
        # [[-gradient1 * input1, -gradient1 *input2, .. k], [-gradient2 * input1, -gradient2 * input2, ... k], ... n]
        if len(curr_gradient) != len(layer.outputs):
            raise ValueError("Invalid gradient size")
        if len(curr_gradient) != len(layer.inputs):
            raise ValueError("Invalid gradient size")
        
        return np.dot(curr_gradient.transpose(), layer.inputs) / len(curr_gradient) # n x k (n neurons, k inputs)
    
    def calc_gradient_biases(self, curr_gradient):
        return np.mean(curr_gradient, axis=0) # n x 1

    # input_batch -> [[input1, input2, ...], [input1, input2, ...], ...]
    def forward_pass(self, input_batch):
        inputs = input_batch  # [[input1, input2, ...], [input1, input2, ...], ...]
        for layer in self.layers:
            inputs = layer.forward(inputs)
            inputs = self.activation_function.forward(inputs)
        inputs = self.final_layer.forward(inputs)
        outputs = self.final_activation_function.forward(inputs)
        return outputs

    
    def calculate_loss(self, output_batch, target_batch):
        ## cross entropy loss
        # mini batch avg loss
        epsilon = 1e-15
        output_batch = np.clip(output_batch, epsilon, 1 - epsilon)
        return np.mean(-np.sum(target_batch * np.log(output_batch), axis=1))

    def train(self, input_batch, target_batch):
        patience = 30
        best_loss = np.inf
        epochs_without_improvement = 0
        max_epochs = 20
        learning_rate = 0.001
        min_loss_threshold = 0.0018
        for epoch in range(max_epochs):
            print(f"training epoch: {epoch}")
            output_batch = self.forward_pass(input_batch)
            if np.any(np.isnan(output_batch)):
                print("NaN detected in outputs!")
                break
            loss = self.calculate_loss(output_batch, target_batch)

            if np.isnan(loss):
                print("NaN detected in loss!")
                break
                
            if loss < min_loss_threshold:
                break

            print(f"epoch: {epoch} loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print("Early stopping!")
                break

            grad_clip_value = 1.0
            start_gradient = output_batch - target_batch
            start_gradient = np.clip(start_gradient, -grad_clip_value, grad_clip_value)

            prev_layer = self.final_layer
            prev_gradient = start_gradient


            if np.any(np.isnan(prev_gradient)):
                print("NaN detected in gradients!")
                break
                
            gradient_weights = self.calc_gradient_weights(prev_gradient, prev_layer)
            gradient_biases = self.calc_gradient_biases(prev_gradient)

            gradient_weights = np.clip(gradient_weights, -grad_clip_value, grad_clip_value)
            gradient_biases = np.clip(gradient_biases, -grad_clip_value, grad_clip_value)

            prev_layer.update_weights(learning_rate, gradient_weights)
            prev_layer.update_biases(learning_rate, gradient_biases)
            for layer in reversed(self.layers):
                prev_gradient = self.calc_gradient_layer(prev_gradient, layer, prev_layer)
                prev_gradient = np.clip(prev_gradient, -grad_clip_value, grad_clip_value)
                if np.any(np.isnan(prev_gradient)):
                    print("NaN detected in hidden layer gradients!")
                    break

                gradient_weights = self.calc_gradient_weights(prev_gradient, layer)
                gradient_biases = self.calc_gradient_biases(prev_gradient)
                
                gradient_weights = np.clip(gradient_weights, -grad_clip_value, grad_clip_value)
                gradient_biases = np.clip(gradient_biases, -grad_clip_value, grad_clip_value)

                layer.update_weights(learning_rate, gradient_weights)
                layer.update_biases(learning_rate, gradient_biases)
                prev_layer = layer
        return best_loss
    
    def full_train(self, x_train, y_train, batch_size):
        best_loss = np.inf
        for i in range(0, len(x_train), batch_size):
            input_batch = x_train[i:i+batch_size]
            target_batch = y_train[i:i+batch_size]
            loss = self.train(input_batch, target_batch)
            if loss < best_loss:
                best_loss = loss
