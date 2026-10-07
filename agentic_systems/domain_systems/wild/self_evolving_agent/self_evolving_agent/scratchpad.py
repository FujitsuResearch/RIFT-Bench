
def binomial_coefficient(n, k):
    """Calculates the binomial coefficient ("n choose k") for non-negative integer inputs `n` and `k`, 
    returning the number of ways to choose `k` items from `n` items without repetition."""
    print("\n\n\nbinomial_coeff TOOL USED")
    if not isinstance(n, int) or not isinstance(k, int):
        return 'Inputs must be integers.'
    if n < 0 or k < 0:
        return 'Inputs must be non-negative.'
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result