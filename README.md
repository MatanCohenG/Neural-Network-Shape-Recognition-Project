### Neural Network for Shape Recognition - README

#### Overview
This repository contains a Python implementation of a neural network designed to classify shapes (triangles, rectangles, and trapezoids) based on images. The network architecture is implemented using numpy and PIL libraries.

#### Files

- **neural_network.py**: Contains the implementation of the neural network class (`NeuralNetwork`) including methods for training (`train`), testing (`test`), backpropagation (`backPropagate`), and utility functions for image loading and dataset creation.

- **README.md**: This file provides an overview of the project, instructions on how to set up the environment, and details on how to run the code.

- **triangles/**, **rectangles/**, **trapezoids/**: Folders containing images of triangles, rectangles, and trapezoids respectively. These are used to create the training and testing datasets.

#### Requirements

To run the code, you need the following dependencies:

- Python 3.x
- numpy
- Pillow (PIL)

#### Setup and Execution

1. **Clone the Repository**:
   ```
   Note: The repository URL will be updated once the project is uploaded to GitHub.
   **need to update later** : git clone https://github.com/your_username/shape-recognition-neural-network.git
   cd shape-recognition-neural-network
   ```

2. **Install Dependencies**:
   ```
   pip install numpy pillow
   ```

3. **Prepare Image Datasets**:
   - Ensure you have separate folders (`triangles/`, `rectangles/`, `trapezoids/`) containing grayscale images of the respective shapes.

4. **Run the Code**:
   - Execute the main script `neural_network.py`:
     ```
     python neural_network.py
     ```
   - The script will train the neural network on the provided image datasets and then test its accuracy on both the training and unseen test sets.

#### Customization

- **Training and Testing**: You can customize the number of training and testing sets by modifying `training_sizes` and `test_size` in the `main` function of `neural_network.py`.
  
- **Network Architecture**: The number of hidden layers in the neural network can be adjusted by passing different values to the `NeuralNetwork` constructor in the `main` function.

#### Results

- After training and testing, the script will display the accuracy of the neural network on both the training and testing sets. It will output predictions and accuracy metrics for each type of shape (triangle, rectangle, trapezoid).

#### Conclusion

This project demonstrates how to implement a simple neural network for image classification using Python. By following the instructions above, you can replicate the experiment, modify parameters, and extend the functionality as needed.

For any questions or issues, feel free to contact [your email/username].

--- 

Adjust the README content as per your actual implementation details and preferences.