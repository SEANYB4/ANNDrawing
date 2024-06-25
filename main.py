import numpy as np
import tkinter as tk

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.random.rand(hidden_size)
        self.bias2 = np.random.rand(output_size)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, x):
        self.hidden = self.sigmoid(np.dot(x, self.weights) + self.bias1)
        output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return output
    

    
class NetworkVisualizer(tk.Tk):

    def __init__(self, neural_network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = neural_network
        self.title("Neural Network Visualizer")
        self.canvas = tk.Canvas(self, width=600, height=400, bg='white')
        self.canvas.pack()
        self.draw_network()


    def draw_network(self):
        layer_configs = [len(self.nn.weights1), len(self.nn.weights2[0]), len(self.nn.bias2)]
        x_spacing = 600 / (len(layer_configs) + 1)
        y_spacing = 400 / (max(layer_configs) + 1)

        # Nodes
        node_positions = {}

        for layer_index, layer_size in enumerate(layer_configs):
            x = (layer_index + 1) * x_spacing
            for node_index in range(layer_size):
                y = (node_index + 1) * y_spacing
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="sky blue")
                node_positions[(layer_index, node_index)] = (x, y)


        # Connections
        for i, (start_layer, end_layer) in enumerate([(0, 1), (1, 2)]):
            for start_node in range(layer_configs[start_layer]):
                for end_node in range(layer_configs[end_layer]):
                    start_x, start_y = node_positions[(i, start_node)]
                    end_x, end_y = node_positions[(i + 1, end_node)]
                    self.canvas.create_line(start_x, start_y, end_x, end_y, fill="gray")


def main():
    nn = NeuralNetwork(4, 64, 5)
    app = NetworkVisualizer(nn)
    app.mainloop()


if __name__ == "__main__":
    main()

