import random
import numpy as np
from PIL import Image
import os

"""
Introduction to Neural Network Shape Recognition Project

The code was authored by the student Matan Cohen.

Project Overview:
This project focuses on the problem of geometric shape recognition using neural networks. Specifically, we are tasked with identifying three shapes: triangles, rectangles, and trapezoids. The goal is to train neural networks to accurately classify these shapes under varying conditions of size and position within a 10x10 matrix.

Methodology:
We have created 20 sets of shapes, each set containing a triangle, a rectangle, and a trapezoid. These shapes vary in size and position within the 10x10 grid. We implemented two neural network architectures:
1. A network with one hidden layer
2. A network with two hidden layers

Both networks were trained using both sequential and random input patterns. We chose to stop the training algorithm when it reached 70-80% accuracy or when it reached a maximum of 20,000 iterations, as implemented in our code.

The project explores three different training set sizes:
1. 4 sets (20% of the data)
2. 9 sets (45% of the data)
3. 15 sets (75% of the data)

For each scenario, we test the networks' performance on the remaining data that was not used in training.



"""

# sigmoid function
def sigmoid(y):
    """
       Computes the sigmoid function for the given input.

       The sigmoid function is used to normalize values to a range between 0 and 1.
    """
    return 1 / (1 + np.exp(-y))


def dsigmoid(y):
    """
       Computes the derivative of the sigmoid function.

       Used in the learning process of the neural network.
    """
    return y * (1.0 - y)


def save_image(data, filename, size=(10, 10)):
    """Saves a data array as an image."""
    image = Image.fromarray((data.reshape(size) * 255).astype(np.uint8))
    image.save(filename)


random.seed(42)  # Set a fixed seed for reproducibility


# Returns a random number between a and b
def rand(a, b):
    return random.uniform(a, b)


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def makeMatrix(I, J, fill=0.0):
    return [[fill] * J for _ in range(I)]


class NeuralNetwork:
    def __init__(self, ni, nh, no):
        """
         Initializes the neural network.

         Parameters:
         ni (int): Number of neurons in the input layer.
         nh (int): Number of neurons in the hidden layer.
         no (int): Number of neurons in the output layer.
         """
        self.ni = ni + 1  # +1 for bias
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        """
                Updates the neural network with new inputs and computes the output.

                Parameters:
                inputs (list): List of input values.

                Returns:
                list: List of computed output values.
        """
        if len(inputs) != self.ni - 1:
            raise ValueError('Input size mismatch')

        # Input nodes
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # Hidden nodes
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # Output nodes
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        """
               Performs the backpropagation learning algorithm.

               Parameters:
               targets (list): List of target values.
               N (float): Learning rate.
               M (float): Momentum.

               Returns:
               float: The computed error value.
        """
        if len(targets) != self.no:
            raise ValueError('Output size mismatch')

        # Calculate output deltas
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # Calculate hidden deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # Update weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # Calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        print('Group size =', len(patterns[0][0]), '| Hidden layers =', self.nh)
        print("0=Rectangle, 0.5=Triangle, 1=Trapezoid")
        for p in patterns:
            predict = np.mean(self.update(p[0]))
            percentile = 100 - round(abs(p[1][0] - predict) * 100, 2)
            print(p[1], 'Result:', predict, 'Accuracy:', percentile, '%')

    def train(self, patterns, iterations=20000, N=0.5, M=0.1, target_accuracy=0.80):
        old_error = float('inf')
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error += self.backPropagate(targets, N, M)
            if i % 1000 == 0:
                fix = round(np.mean(error), 6)
                print(f'Iterations={i}, Error={fix}, Improvement={round(old_error - fix, 5)}')
                old_error = fix
                self.test(patterns)
                if old_error <= (1 - target_accuracy):
                    print(f"Target accuracy ({target_accuracy * 100}%) achieved. Stopping training.")
                    break


def load_images_from_folder(folder_path, shape_size=10):
    """
        Loads images from a specified folder and converts them to one-dimensional arrays (flattened matrices).

        Process:
        1. Reads each image file in the given folder.
        2. Resizes each image to (shape_size x shape_size) pixels.
        3. Converts each image to grayscale.
        4. Flattens the 2D matrix into a 1D array.
        5. Normalizes pixel values to a range between 0 and 1.

        Parameters:
        folder_path (str): The path to the folder containing the images.
        shape_size (int): The desired size of the images (default: 10).

        Returns:
        list: A list of numpy 1D arrays, each representing an image.
    """
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((shape_size, shape_size))
        img_data = np.array(img.convert('L')).flatten() / 255.0
        images.append(img_data)
    return images


def create_dataset(num_sets, triangles, rectangles, trapezoids):
    dataset = []
    for _ in range(num_sets):
        shape_type = random.choice(['triangle', 'rectangle', 'trapezoid'])
        if shape_type == 'triangle':
            dataset.append([random.choice(triangles), [0.5]])
        elif shape_type == 'rectangle':
            dataset.append([random.choice(rectangles), [0]])
        elif shape_type == 'trapezoid':
            dataset.append([random.choice(trapezoids), [1]])
    return dataset


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    base_path = "C:\\Users\\Matan Cohen\\Documents\\FINAL_IRINA"

    triangles_path = os.path.join(base_path, "triangles")
    rectangles_path = os.path.join(base_path, "rectangles")
    trapezoids_path = os.path.join(base_path, "trapezoids")

    triangles = load_images_from_folder(triangles_path)
    rectangles = load_images_from_folder(rectangles_path)
    trapezoids = load_images_from_folder(trapezoids_path)

    # Define training and testing sets sizes
    training_sizes = [4, 9, 15]

    for train_size in training_sizes:
        test_size = 20 - train_size
        train_set = create_dataset(train_size, triangles, rectangles, trapezoids)
        test_set = create_dataset(test_size, triangles, rectangles, trapezoids)

        for hidden_layers in [1, 2]:
            print(f"\n--- Training with {train_size} sets and {hidden_layers} hidden layer(s) ---")
            n = NeuralNetwork(len(train_set[0][0]), hidden_layers, 1)
            n.train(train_set, iterations=20000, target_accuracy=0.80)

            print("\n--- Testing on trained set (sequential) ---")
            n.test(train_set)
            print("\n--- Testing on trained set (random) ---")
            random.shuffle(train_set)
            n.test(train_set)

            print("\n--- Testing on unseen set (sequential) ---")
            n.test(test_set)
            print("\n--- Testing on unseen set (random) ---")
            random.shuffle(test_set)
            n.test(test_set)


if __name__ == "__main__":
    main()
