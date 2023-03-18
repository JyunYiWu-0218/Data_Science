##Vector 較重要的測試及結果(示意, 不是 Python 正確語法)
---

Tests with two dimensions::  
        >>>v1_clone = eval(repr(v1))    
        >>>v1 == v1_clone    
                True    
        >>> octets = bytes(v1)  
        >>> octets  
                b'd\\x00\\x00\\x00\\x00\\x00\\x00\\x08@\\x00\\x00\\x00\\x00\\x00\\x00\\x10@'
        >>> bool(v1), bool(Vector(0, 0))  
                (True, False)
---
    
Test of ".frombytes()" class method:  
        >>> v1_clone = Vector.frombytes(bytes(v1))  
        >>> v1_clone  
                Vector(3.0, 4.0)  
        >>> v1 == v1_clone
                True  

---
Tests with three dimensions:  
        >>> v1 = Vector(3, 4, 5)  
        >>> x, y, z = v1  
        >>> x, y, z  
                (3.0, 4.0, 5.0)  
        >>> v1_clone = eval(repr(v1))  
        >>> v1 == v1_clone  
                True  
        >>> bool(v1), bool(Vector(0, 0, 0))  
                (True, False)  

---
Tests of sequence behavior:
        >>> v1 = Vector(3, 4, 5)  
        >>> v1(0), v1(len(v1)-1), v1(-1)  
                (3.0, 5.0, 5.0)  

Test of slicing:
        >>> v7 = Vector(range(7))  
        >>> v7(-1)  
                6.0  
        >>> v7(1:4)  
                Vector(1.0, 2.0, 3.0)  
        >>> v7(1,2)  
                ==Traceback (most recent call last):==      
                ==...==  
                ==TypeError: Vector indices must be integers==  

---
Tests of dynamic attribute access:
        >>> v7 = Vector(range(10))  
        >>> v7.x  
                0.0  
        >>> v7.y, v7.z, v7.t  
                (1.0, 2.0, 3.0)

---

Dynamic attribute lookup failures:  
        >>> v7.k    
                ***Traceback (most recent call last):***    
                ***...***    
                ***AttributeError: 'Vector' object has no attribute 'k'***    
        >>> v3 = Vector(range(3))  
        >>> v3.t  
                ***Traceback (most recent call last):***    
                ***...***    
                ***AttributeError: 'Vector' object has no attribute 't'***    
        >>> v3.spam  
                ***Traceback (most recent call last):***    
                ***...***    
                ***AttributeError: 'Vector' object has no attribute 'spam'***    
---

Tests of hashing::  
        >>> v1 = Vector(3, 4)    
        >>> v3 = Vector(3, 4, 5)     
        >>> v6 = Vector(range(6))    
        >>> hash(v1), hash(v3), hash(v6)    
                (7, 2, 1)    

---

Most hash values of non-integers vary from a 32-bit to 64-bit CPython build:  
        >>> import sys   
        >>> hash(v2) == (384307168202284039 if sys.maxsize > 2^32 else 357915986)   
                True  
---

Tests of "format()" with spherical coordinates in 2D, 3D and 4D:    
        >>> format(Vector(1, 1), 'h') doctest:+ELLIPSIS  
                '<1.414213..., 0.785398...>'  
        >>> format(Vector(1, 1), '.3eh')  
                '<1.414e+00, 7.854e-01>'  
        >>> format(Vector(1, 1), '0.5fh')  
                '<1.41421, 0.78540>'  
        >>> format(Vector(1, 1, 1), 'h') # doctest:+ELLIPSIS  
                '<1.73205..., 0.95531..., 0.78539...>'  
        >>> format(Vector(2, 2, 2), '.3eh')  
                '<3.464e+00, 9.553e-01, 7.854e-01>'  
        >>> format(Vector(0, 0, 0), '0.5fh')  
                '<0.00000, 0.00000, 0.00000>'  
        >>> format(Vector([-1, -1, -1, -1]), 'h') doctest:+ELLIPSIS  
                '<2.0, 2.09439..., 2.18627..., 3.92699...>'  
        >>> format(Vector(2, 2, 2, 2), '.3eh')  
                '<4.000e+00, 1.047e+00, 9.553e-01, 7.854e-01>'  
        >>> format(Vector(0, 1, 0, 0), '0.5fh')  
                '<1.00000, 1.57080, 0.00000, 0.00000>'  



