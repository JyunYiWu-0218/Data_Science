from array import array
import reprlib
import math
import numbers
import functools
import operator
import itertools


class Vector:
    typecode = 'd'
    
    def __init__(self, components):
        self._components  = array(self.typecode, components)
    
    def __iter__(self):
        return iter(self._components)
    
    def __repr__(self):
        components = reprlib.repr(self._components) 
        # repr(): A class can control what this function returns for its instances by defining a __repr__() method. If sys.displayhook() is not accessible(??), this function will raise RuntimeError.
        # reprlib.repr(): size limits for different object types are added to avoid the generation of representations which are excessively long.
        components = components[components.find('['):-1] 
        # The find() method returns -1 if the value is not found.
        # string.find(value, start, end)
        return 'Vector({})'.format(components)
    
    def __str__(self):
        return str(tuple(self))
    
    def __bytes__(self):
        return (bytes([ord(self._components)]) + # The ord() function returns an integer representing the "Unicode" character.
                bytes(self._components)) # bytes([source[, encoding[, errors]]])
                # bytes() method returns a bytes object which is an immutable (cannot be modified) sequence of integers in the range 0 <=x < 256.
                # bytes() Parameters(bytes() takes three optional parameters):
                # 1.source (Optional) - source to initialize the array of bytes.
                # 2.encoding (Optional) - if the source is a string, the encoding of the string.
                # 3.errors (Optional) - if the source is a string, the action to take when the encoding conversion fails.
    
    def __eq__(self, other):
        return (len(self) == len(other) and
                all(a == b for a, b in zip(self, other)))
    
    def __hash__(self):
        Hashes = (hash(x) for x in self) # The hash() method returns the hash value of an object.
        return functools.reduce(operator.xor, Hashes, 0) 
        # functools.reduce(function, iterable[, initializer]),  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5).
        # The functools.reduce(fun,seq) is used to apply a particular function passed in its argument to all of the list elements mentioned in the sequence passed along.
        
    def __abs__(self):
        return math.sqrt(sum(x * x for x in self))
    
    def __bool__(self):
        return bool(abs(self))
    
    def __len__(self):
        return len(self._components)
    
    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        
        else:
            msg = '{.__name__} indices must be integers'
            raise TypeError(msg.format(cls))
            
    shortcut_names = 'xyzt'
    
    def __getattr__(self, name):
        cls = type(self)
        if len(name) == 1 :
            pos = cls.shortcut_names.find(name)
            
            if 0 <= pos < len(self._components):
                return self._components[pos]
        msg = '{.__name__ !r} indices must be integers {!r}'
        raise AttributeError(msg.format(cls, name))
        
    def angle(self, n):
        r = math.sqrt(sum(x * x for x in self[n:]))
        a = math.atan2(r, self[n - 1]) # The math.atan2() method returns the arc tangent of y/x, in radians.(The returned value is between PI and -PI.)
        if (n == len(self) - 1) and (self[-1] < 0):
            return math.pi * 2 - a
        else:
            return a
        
        
    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))
    
    def __format__(self, fmt_spec = ''):
        
        if fmt_spec.endswith('h'): # string.endswith(value, start, end): The endswith() method returns True if the string ends with the specified value, otherwise False.
            fmt_spec = fmt_spec[:-1]
            coords = itertools.chain([abs(self)], self.angles()) # chain (*iterables): It groups all the iterables together and produces a single iterable as output.
            outer_fmt = '<{}>'
            
        else:
            
            coords = self
            outer_fmt = '({})' 
            components = (format(c, fmt_spec) for c in coords) 
            return outer_fmt.format(', '.join(components)) 
            # string.join(iterable): The join() method takes all items in an iterable and joins them into one string.(A string must be specified as the separator.)
        
    @classmethod
    def frombytes(cls, octets):
            
            typecode = chr(octets[0]) # The chr() method converts an integer to its unicode character and returns it.
            memv = memoryview(octets[1:]).cast(typecode)
            return cls(memv)

        
