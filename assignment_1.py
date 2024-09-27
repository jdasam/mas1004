from math import sin, cos, exp
from typing import List, Tuple, Union
import torch
import torch.nn as nn



class MyFunction:
  def __init__(self, params: List[float]) -> None:
    '''
    The instance is initialized with a list of parameters.
    The parameters are in the following order:
    [a, b, c, d, e]
    where the function is:
    $f(x) =  ax + b\cos(x) + c\sin(x) + dx^2 + e $
    '''
    self.params = params

  def __len__(self) -> int:
    '''
    return the number of NON-ZERO parameters.
    For example, if the parameters are [1, 0, 3, 0, 5], then the length is 3, ignoring the two zeros.

    TODO: implement this
    '''
    return

  def __add__(self, other: 'MyFunction') -> 'MyFunction':
    '''
    This function adds two MyFunction objects together.
    It returns a new MyFunction object with the sum of the parameters.
    It is called when you do something like:
    >>> f1 = MyFunction([1, 2, 3])
    >>> f2 = MyFunction([4, 5, 6])
    >>> f3 = f1 + f2
    >>> print(f3.params)
    [5, 7, 9]

    TODO: implement this
    '''
    return

  def __call__(self, x:float) -> float:
    '''
    return the function value at x
    $f(x) =  ax + b\cos(x) + c\sin(x) + dx^2 + e $

    TODO: implement this
    '''
    return

  def calculate_list_input(self, x: List[float]) -> List[float]:
    '''
    args:
      x: a list of floats
    return:
      a list of function values at x

    TODO: implement this using __call__
    HINT: you can use self(float_value) to call __call__
    '''
    return


def answer_for_problem2()-> Tuple[MyFunction, MyFunction, MyFunction]:
  '''
  args:
    None

  returns:
    function_a: MyFunction
    function_b: MyFunction
    function_c: MyFunction
  '''

  return


class AnotherFunction(MyFunction):
  def __init__(self, params: List[float]) -> None:
    super().__init__(params)

  def __call__(self, x:float) -> float:
    '''
    args:
      x: a float value
    return:
      a float value
      $f(x) = ax + bx^3 + c\times2^x + d\times3^x + e \times \exp(x)$

    TODO: implement this
    '''
    return


def matrix_multiply_using_for_loop(mat_a: List[List[float]], mat_b: List[List[float]]) -> list:
  '''
  args:
    mat_a: List of list of float (or integer), has shape m x n. If not a matrix, return 'Error'
    mat_b: List of list of float (or integer), has shape n x p. If not a matrix, return 'Error'
  return:
    an m x p matrix in list of list format

  TODO: implement this function using for loop and check_condition()
  '''
  def check_condition():
    '''
    This is inner function. You do not need to get arguments from outside.
    In other words, variables defined outside of this function is available inside this function.
    For example, you can refer to mat_a and mat_b inside this function, without passing them as arguments.

    args:
      None
    return:
      True if the mat_a and mat_b can be represented as a matrix and can be multiplied, False otherwise.
    '''

    # TODO: implement this function
    return

  condition = check_condition()
  if condition is False:
    return 'Error'

  return


def add_tensor(ten_a, ten_b):
  '''
  args:
    ten_a: a tensor
    ten_b: a tensor
  return:
    a tensor or 'Error'

  TODO: implement this
  If two tensors cannot be added, return 'Error'
  '''

  condition = True # replace this line with correct logic


  if condition is False:
    return 'Error'

  return


def elementwise_mul_tensor(ten_a, ten_b):
  '''
  args:
    ten_a: a 2D tensor
    ten_b: a 2D tensor
  return:
    a tensor or 'Error'

  TODO: implement this
  If two tensors cannot be added, return 'Error'
  '''

  condition = True # replace this line with correct logic

  if condition is False:
    return 'Error'

  return

def tensor_matrix_mutiplication(ten_a, ten_b):
  '''
  args:
    ten_a: a 2D tensor
    ten_b: a 2D tensor
  return:
    a tensor or 'Error'

  TODO: implement this
  If two tensors cannot be added, return 'Error'
  '''

  condition = True # replace this line with correct logic

  if condition is False:
    return 'Error'

  return


def answer_to_problem_34(input_tensor):
  '''
  args:
    input_tensor: a 3D tensor of shape (2, 3, 4)
    >>> tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
  return:
    a tuple of tensors
    first_tensor: a 3D tensor of shape (2, 2, 2)
    >>> tensor([[[ 5,  6],
                [ 9, 10]],

                [[17, 18],
                [21, 22]]])
    second_tensor: a 3D tensor of shape (1, 1, 1)
    >>> tensor([[[19]]])
    third_tensor: a 2D tensor of shape (2, 3)
    >>> tensor([[ 4,  5,  6],
                [16, 17, 18]])

  TODO: complete this function using tensor slicing
  '''

  return output_a, output_b, output_c


def answer_to_problem_35(problem):
  '''
  args:
    problem: a tensor of shape (2, 3, 4)
    >>> tensor([[[111, 112, 113, 114],
         [121, 122, 123, 124],
         [131, 132, 133, 134]],

        [[211, 212, 213, 214],
         [221, 222, 223, 224],
         [231, 232, 233, 234]]])

  return:
    a tensor
    >>> tensor([[111, 211, 121, 221, 131, 231, 112, 212, 122, 222, 132, 232],
          [113, 213, 123, 223, 133, 233, 114, 214, 124, 224, 134, 234]])

  TODO: complete this function using one permute and one reshape operation.
  '''
  return


def get_l2_norm_of_vector(ten_a: torch.Tensor) -> torch.Tensor:
  '''
  args:
    ten_a: a tensor of ndim == 1
  return:
    l2 norm of the ten_a
    l2 norm is defined as:
    $||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}$

  TODO: implement this
  '''
  assert ten_a.ndim == 1

  return

def normalize_vector(ten_a: torch.Tensor) -> torch.Tensor:
  '''
  args:
    ten_a: a tensor of ndim == 1
  return:
    a tensor that is normalized version of ten_a
    normalized version of ten_a is defined as:
    $x_{normalized} = \frac{x}{||x||_2}$
  TODO: implement this using get_l2_norm_of_vector
  '''
  assert ten_a.ndim == 1
  return

def get_cosine_similarity_of_two_vectors(vec_a, vec_b):
  '''
  args:
    vec_a: a tensor of ndim == 1
    vec_b: a tensor of ndim == 1
  return:
    cosine similarity of vec_a and vec_b
    cosine similarity is defined as:
    $cos(\theta) = \frac{a \cdot b}{||a||_2 ||b||_2}$
  TODO: Implement this using normalize_vector and torch.dot
  '''
  assert vec_a.ndim == 1
  assert vec_b.ndim == 1

  return

def normalize_tensor(mat:torch.Tensor, dim:int):
  '''
  args:
    mat: a tensor of arbitrary shape
    dim: the dimension to normalize
  return:
    a tensor that is normalized version of mat
    among the dimensions,
    normalized version of mat is defined as:
    $x_{normalized} = \frac{x}{||x||_2}$
  '''
  assert dim < mat.ndim

  # TODO: implement this without using torch.norm function
  # You can use torch.sum(), torch.sqrt()

  return

def get_cosine_similarity_of_two_matrices_by_for_loop(mat_a, mat_b):
  '''
  args:
    mat_a: a 2D tensor
    mat_b: a 2D tensor
  return:
    cosine similarity of mat_a and mat_b
    cosine similarity is defined as:
    $cos(\theta) = \frac{a \cdot b}{||a||_2 ||b||_2}$

  TODO: Compute cosine similarity of two matrices using for loop
  use normalize_tensor() and torch.dot(), or
  '''
  assert mat_a.ndim == 2
  assert mat_b.ndim == 2
  assert mat_a.shape[1] == mat_b.shape[1]

  return

def get_consine_similarity_of_two_matrices(mat_a: torch.Tensor, mat_b: torch.Tensor):
  '''
  args:
    mat_a: a tensor of two dimensions
    mat_b: a tensor of two dimensions
    dim: the dimension to calculate cosine similarity
  return:
    cosine similarity between every vector of mat_a and mat_b
    cosine similarity is defined as:
    $cos(\theta) = \frac{a \cdot b}{||a||_2 ||b||_2}$
  TODO: Implement this using normalize_tensor() and torch.mm()
  '''
  assert mat_a.ndim == 2
  assert mat_b.ndim == 2

  return

class MyModel(nn.Module): # inherit from nn.Module
  def __init__(self):
    super().__init__() # call the __init__() of nn.Module

    # TODO: define layers with nn.Linear()
    # Caution: You have to strictly follow the order of layers in declaration
    # So that the assertion code below does not fail.
    self.layer1 = None # replace this line with correct code
    self.layer2 = None # replace this line with correct code
    self.layer3 = None # replace this line with correct code



  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    args:
      x: a 2D tensor
    return:
      a 2D tensor
      Output of the model
      x -> self.layer1 -> ReLU -> self.layer2 -> ReLU -> self.layer3 -> return

    TODO: Implement this using self.layer1, self.layer2, self.layer3, torch.relu()
    CAUTION: Do not forget to use torch.relu() after each layer except the last layer.
    '''
    return
  

if __name__ == '__main__':
  my_function = MyFunction([1, 0, 3, 0, 5])
  test_input = [-1, 0, 3]

  assert len(my_function) == 3, 'your __len__ implementation is wrong'
  assert len(MyFunction([0, 0, 0, 0, 0])) == 0, 'your __len__ implementation is wrong'
  assert len(MyFunction([2, 3, 2, -2, -2])) == 5, 'your __len__ implementation is wrong'

  assert my_function(0) == 5, 'your __call__ implementation is wrong'
  assert my_function(-1) == 1.4755870455763107, 'your __call__ implementation is wrong'

  function_1 = MyFunction([1, 0, 3, 0, 5])
  function_2 = MyFunction([-2, 1, 0, -1, 2])

  function_3 = function_1 + function_2
  assert isinstance(function_3, MyFunction), 'MyFunction.__add__ has to return a MyFunction object'
  assert function_3.params == [-1, 1, 3, -1, 7], 'your __add__ implementation is wrong'

  test_output = function_1.calculate_list_input(test_input)
  assert test_output == [1.4755870455763107, 5.0, 8.423360024179601], 'your calculate_list_input implementation is wrong'

  func_a, func_b, func_c = answer_for_problem2()

  assert isinstance(func_a, MyFunction) and isinstance(func_b, MyFunction) and isinstance(func_c, MyFunction), "Please return three MyFunction objects"


  x_examples = [(i-100)/20 for i in range(201)]

  y_a = func_a.calculate_list_input(x_examples)
  y_b = func_b.calculate_list_input(x_examples)
  y_c = func_c.calculate_list_input(x_examples)

  assert torch.allclose(torch.tensor(y_a[10:15]), torch.tensor([-8.0, -7.9, -7.8, -7.7, -7.6]), atol=1e-6), "Function a is not correct"
  assert torch.allclose(torch.tensor(y_b[10:15]), torch.tensor([-18.0691, -17.8043, -17.5431, -17.2852, -17.0304]), atol=1e-4), "Function a is not correct"
  assert torch.allclose(torch.tensor(y_c[10:15]), torch.tensor([1.3227, 1.1534, 0.9812, 0.8066, 0.6299]), atol=1e-4), "Function a is not correct"

  another_func = AnotherFunction([3, 0.1, -0.01, -0.01, 0.02])

  # You do not have to implement calculate_list_input() again. It is inherited from MyFunction
  y_another = another_func.calculate_list_input(x_examples)

  assert torch.allclose(torch.tensor(y_another[30:35]), torch.tensor([-14.7880, -14.4569, -14.1309, -13.8101, -13.4942]), atol=3e-4), "Your implementation of AnotherFunction is not correct"
  test_a = [[1, 2, 3], [4, 5]]
  test_b = [[1, 2], [3, 4], [5, 6]]
  test_c = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

  result_a = matrix_multiply_using_for_loop(test_a, test_b)
  result_b = matrix_multiply_using_for_loop(test_b, test_c)
  result_c = matrix_multiply_using_for_loop(test_c, test_b)

  assert result_a == 'Error', f"result_a should be 'Error', but got {result_a}"
  assert result_b == 'Error', f"result_b should be 'Error', but got {result_b}"
  assert result_c == [[22, 28], [49, 64], [76, 100]], f"result_c should be [[22, 28], [49, 64], [76, 100]], but got {result_c}"
  tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
  tensor_b = torch.tensor([[2, 4, 3], [6, 6, 7]])
  result = add_tensor(tensor_a, tensor_b)
  tensor_c = torch.randn(3, 3)
  tensor_d = torch.randn(2, 4)
  tensor_e = torch.randn(2, 3, 1)
  tensor_f = torch.randn(2, 3)
  assert (result == torch.tensor([[3, 6, 6], [10, 11, 13]])).all(), f"result should be [[3, 6, 6], [10, 11, 13]], but got {result}"
  assert add_tensor(tensor_a, tensor_c) == 'Error', f"result should be 'Error', but got {add_tensor(tensor_a, tensor_c)}"
  assert add_tensor(tensor_a, tensor_d) == 'Error', f"result should be 'Error', but got {add_tensor(tensor_a, tensor_d)}"
  assert add_tensor(tensor_a, tensor_e) == 'Error', f"result should be 'Error', but got {add_tensor(tensor_a, tensor_e)}"
  assert add_tensor(tensor_a, tensor_f) != 'Error', f"result should not be 'Error', but got {add_tensor(tensor_a, tensor_f)}"
  tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
  tensor_b = torch.tensor([[2, 4, 3], [6, 6, 7]])

  result = elementwise_mul_tensor(tensor_a, tensor_b)

  tensor_c = torch.randn(3, 3)
  tensor_d = torch.randn(2, 4)
  tensor_e = torch.randn(2, 3, 1)
  tensor_f = torch.randn(2, 3)

  assert (result == torch.tensor([[2, 8, 9], [24, 30, 42]])).all(), f"result should be [[2, 8, 9], [24, 30, 42]], but got {result}"
  assert elementwise_mul_tensor(tensor_a, tensor_c) == 'Error', f"result should be 'Error', but got {elementwise_mul_tensor(tensor_a, tensor_c)}"
  assert elementwise_mul_tensor(tensor_a, tensor_d) == 'Error', f"result should be 'Error', but got {elementwise_mul_tensor(tensor_a, tensor_d)}"
  assert elementwise_mul_tensor(tensor_a, tensor_e) == 'Error', f"result should be 'Error', but got {elementwise_mul_tensor(tensor_a, tensor_e)}"
  assert elementwise_mul_tensor(tensor_a, tensor_f) != 'Error', f"result should not be 'Error', but got {elementwise_mul_tensor(tensor_a, tensor_f)}"
  tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
  tensor_b = torch.tensor([[2, 4], [6, 6], [7, 8]])

  result = tensor_matrix_mutiplication(tensor_a, tensor_b)

  tensor_c = torch.randint(low=-5, high=5, size=(3, 5))
  tensor_d = torch.randint(low=-5, high=5, size=(2, 3))
  tensor_e = torch.randint(low=-5, high=5, size=(4, 3))

  assert (result == torch.tensor([[35, 40], [80, 94]])).all(), f"result should be [[35, 40], [80, 94]], but got {result}"
  assert tensor_matrix_mutiplication(tensor_a, tensor_c).shape == torch.Size([2, 5]), f"result should be a tensor of shape (2, 5), but got {tensor_matrix_mutiplication(tensor_a, tensor_c)}"
  assert tensor_matrix_mutiplication(tensor_a, tensor_d) == 'Error', f"result should be 'Error', but got {tensor_matrix_mutiplication(tensor_a, tensor_d)}"
  assert tensor_matrix_mutiplication(tensor_a, tensor_e) == 'Error', f"result should be 'Error', but got {tensor_matrix_mutiplication(tensor_a, tensor_e)}"


  input_tensor = torch.arange(24).reshape(2, 3, 4)

  desired_output_a = torch.tensor([[[5, 6], [9, 10]], [[17, 18], [21, 22]]])
  desired_output_b = torch.tensor([[[19]]])
  desired_output_c = torch.tensor([[ 4,  5,  6], [16, 17, 18]])
  output_a, output_b, output_c = answer_to_problem_34(input_tensor)

  assert (output_a == desired_output_a).all(), f"Your implementation is wrong. Please try again."
  assert (output_b == desired_output_b).all(), f"Your implementation is wrong. Please try again. Pay attention to the shape of the tensor."
  assert (output_c == desired_output_c).all(), f"Your implementation is wrong. Please try again."

  atensor = torch.tensor([[[111, 112, 113, 114],
                          [121, 122, 123, 124],
                          [131, 132, 133, 134]],
                          [[211, 212, 213, 214],
                          [221, 222, 223, 224],
                          [231, 232, 233, 234]]])

  result = answer_to_problem_35(atensor)
  assert (result==torch.tensor([[111, 211, 121, 221, 131, 231, 112, 212, 122, 222, 132, 232],
                                [113, 213, 123, 223, 133, 233, 114, 214, 124, 224, 134, 234]])).all() , f"Your answer is wrong. Please try again. Your answer is {result}"

  torch.manual_seed(0)
  ten_a = torch.rand(10)
  ten_b = torch.rand(20)

  assert torch.isclose(get_l2_norm_of_vector(ten_a), torch.tensor(1.73478), atol=1e-4), 'Your answer is wrong'
  assert torch.isclose(get_l2_norm_of_vector(ten_b), torch.tensor(2.38497), atol=1e-4), 'Your answer is wrong'
  torch.manual_seed(0)
  ten_a = torch.rand(10)
  ten_b = torch.rand(20)

  assert abs(get_l2_norm_of_vector(normalize_vector(ten_a)) - 1) < 1e-4, 'The length of the normalized vector should be 1'
  assert abs(get_l2_norm_of_vector(normalize_vector(ten_b)) - 1) < 1e-4, 'The length of the normalized vector should be 1'
  torch.manual_seed(0)
  ten_a = torch.rand(10)
  ten_b = torch.rand(10)
  assert torch.isclose(get_cosine_similarity_of_two_vectors(ten_a, ten_b), torch.tensor(0.9352), atol=1e-4), 'Your answer is wrong'

  torch.manual_seed(0)
  ten_a = torch.rand(5)
  ten_b = torch.rand(5)
  assert torch.isclose(get_cosine_similarity_of_two_vectors(ten_a, ten_b), torch.tensor(0.7315), atol=1e-4), 'Your answer is wrong'

  torch.manual_seed(0)
  ten_a = torch.randn(3, 4)
  ten_b = torch.randn(2, 5, 4)

  torch.manual_seed(0)
  ten_a = torch.randn(3, 4)
  ten_b = torch.randn(2, 5, 4)

  answer_a = torch.tensor([[[ 0.7640, -0.1976, -0.9495,  0.5525],
                            [-0.5377, -0.9419,  0.1758,  0.8145],
                            [-0.3566, -0.2716, -0.2600,  0.1769]]])
  answer_b = torch.tensor([[ 0.5615, -0.1069, -0.7939,  0.2071],
                          [-0.5424, -0.6995,  0.2017,  0.4192],
                          [-0.6956, -0.3901, -0.5770,  0.1761]])
  answer_c = torch.tensor([[[ 0.9374, -0.2263,  0.8385,  0.4780],
                            [ 0.2807,  0.6889, -0.1281, -0.0862],
                            [ 0.5333,  0.4403,  0.9733,  0.1581],
                            [ 0.3547, -0.6005, -0.8686, -0.9775],
                            [-0.1983, -0.5543,  0.9701,  0.3067]],

                            [[-0.3484, -0.9741,  0.5449,  0.8784],
                            [-0.9598,  0.7248,  0.9918,  0.9963],
                            [-0.8459,  0.8978, -0.2297,  0.9874],
                            [-0.9350, -0.7997, -0.4955, -0.2109],
                            [-0.9801,  0.8323, -0.2425,  0.9518]]])
  answer_d = torch.tensor([[[ 0.3022, -0.1018,  0.9321,  0.1718],
                            [ 0.1832,  0.9625, -0.1579, -0.1229],
                            [ 0.3662,  0.4947,  0.7879,  0.0211],
                            [ 0.2401, -0.1604, -0.3268, -0.8999],
                            [-0.2178, -0.7493,  0.5678,  0.2620]],

                            [[-0.1371, -0.5349,  0.7394,  0.3852],
                            [-0.2822,  0.4561,  0.5507,  0.6395],
                            [-0.4897,  0.8505, -0.1567,  0.1112],
                            [-0.8789, -0.2965, -0.2588, -0.2695],
                            [-0.6109,  0.6384, -0.0805,  0.4613]]])

  assert torch.allclose(normalize_tensor(ten_a, 0), answer_a, atol=1e-3), 'Your answer is wrong'
  assert torch.allclose(normalize_tensor(ten_a, 1), answer_b, atol=1e-3), 'Your answer is wrong'
  assert torch.allclose(normalize_tensor(ten_b, 0), answer_c, atol=1e-3), 'Your answer is wrong'
  assert torch.allclose(normalize_tensor(ten_b, 2), answer_d, atol=1e-3), 'Your answer is wrong'

  torch.manual_seed(0)
  ten_a = torch.randn(6, 3)
  ten_b = torch.randn(4, 3)

  answer = torch.tensor([[-0.7931,  0.5368,  0.0880,  0.1611],
                        [ 0.3438,  0.0073,  0.4938,  0.4885],
                        [ 0.8885, -0.2307, -0.1930, -0.5425],
                        [-0.6407,  0.2590, -0.2294, -0.1603],
                        [ 0.8627, -0.8801, -0.7024, -0.7978],
                        [ 0.4416, -0.1207, -0.4180, -0.6963]])

  assert torch.allclose(get_cosine_similarity_of_two_matrices_by_for_loop(ten_a, ten_b), answer, atol=1e-3), 'Your answer is wrong'

  torch.manual_seed(0)
  ten_a = torch.randn(6, 3)
  ten_b = torch.randn(4, 3)



  assert torch.allclose(get_consine_similarity_of_two_matrices(ten_a, ten_b), answer, atol=1e-3), 'Your answer is wrong'
  torch.manual_seed(0)
  model = MyModel()
  test_input = torch.randn(5, 1)
  test_input2 = torch.randn(16, 1)
  test_out = model(test_input)
  test_out2 = model(test_input2)
  # Following code is for testing your implementation
  assert hasattr(model, 'layer1'), 'layer1 is not defined'
  assert hasattr(model, 'layer2'), 'layer2 is not defined'
  assert hasattr(model, 'layer3'), 'layer3 is not defined'
  assert isinstance(model.layer1, nn.Linear), 'layer1 should be nn.Linear'
  assert isinstance(model.layer2, nn.Linear), 'layer2 should be nn.Linear'
  assert isinstance(model.layer3, nn.Linear), 'layer3 should be nn.Linear'
  assert model.layer1.in_features == 1, 'layer1 should have 1 input feature'
  assert model.layer1.out_features == 6, 'layer1 should have 6 output features'
  assert model.layer2.in_features == 6, 'layer2 should have 6 input features'
  assert model.layer2.out_features == 4, 'layer2 should have 4 output features'
  assert model.layer3.in_features == 4, 'layer3 should have 4 input features'
  assert model.layer3.out_features == 1, 'layer3 should have 1 output feature'

  assert torch.allclose(test_out, torch.tensor([-0.6696, -0.5911, -0.7009, -0.6649, -0.5941]).unsqueeze(1), atol=1e-4), 'Your forward() is wrong'
  assert torch.allclose(test_out2, torch.tensor([-0.5775, -0.5609, -0.5633, -0.6565, -0.6954, -0.6512, -0.5585, -0.5615,
          -0.6353, -0.6253, -0.5715, -0.6060, -0.6122, -0.6473, -0.6873, -0.6184]).unsqueeze(1), atol=1e-4), 'Your forward() is wrong'
  
  print('All test cases passed')