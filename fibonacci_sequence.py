
def fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number.

    Args:
        n: The position of the Fibonacci number to calculate (must be a positive integer).

    Returns:
        The nth Fibonacci number.

    Raises:
        TypeError: If n is not an integer.
        ValueError: If n is less than 1.
    """
    if not isinstance(n, int):
        raise TypeError("Input 'n' must be an integer.")
    if n < 1:
        raise ValueError("Input 'n' must be greater than or equal to 1.")

    # Base cases
    if n == 1:
        return 1
    if n == 2:
        return 1

    # Iterative calculation for n > 2
    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

# Test with n=10
if __name__ == "__main__":
    print(f"The 10th Fibonacci number is: {fibonacci(10)}")
