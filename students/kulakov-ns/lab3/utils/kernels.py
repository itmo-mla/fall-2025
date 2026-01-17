def kernel_function(kernel_arg: float) -> float:
    if kernel_arg <= 0:
        return 1.0
    if kernel_arg >= 1:
        return 0.0
    return 1.0 / kernel_arg
