import numpy as np
from core.differentiation.finite_difference import create_forward_difference_table


def main():
    """
    Imports the forward difference table generator, generates a table for a
    sample dataset, and prints it.
    """

    # Sample data (you can modify this)
    x = np.array([1, 2, 3, 4])
    f = np.array([1, 8, 27, 64])


    # Generate the table
    try:
        table = create_forward_difference_table(x, f)

        # Print the table
        print("Forward Difference Table:")
        for i, col in enumerate(table):
           print(f"Column {i}: {col}")
    except ValueError as e:
          print(f"Error: {e}")


if __name__ == "__main__":
    main()