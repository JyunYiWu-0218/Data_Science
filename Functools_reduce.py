def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, iterable[, initial]) -> value
    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """

    it = iter(sequence) #  iter(object, sentinel [optional]): The iter() method returns an iterator for the given argument.

    if initial is _initial_missing:
        try:
            value = next(it) # next(iterator, default): The next() function returns the next item from the iterator.(with iter()) 
        except StopIteration:
            raise TypeError("reduce() of empty iterable with no initial value") from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value
