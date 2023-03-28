### Python Buffer Protocol
~  The buffer protocol provides a way to access the internal data of an object. This internal data is a memory array or a buffer.
~  The buffer protocol allows one object to expose its internal data (buffers) and the other to access those buffers without intermediate copying.
~  This protocol is only accessible to us at the C-API level and not using our normal codebase.
~  So, in order to expose the same protocol to the normal Python codebase, memory views are present.

### memory views
~ A memory view is a safe way to expose the buffer protocol in Python.
~ It allows you to access the internal buffers of an object by creating a memory view object.

***Why buffer protocol and memory views are important?***
~ Remember that whenever we perform some action on an object (call a function of an object, slice an array), Python needs to create a copy of the object.
~ If we have large data to work with (eg. binary data of an image), we would unnecessarily create copies of huge chunks of data, which serves almost no use.
~ Using the buffer protocol, we can give another object access to use/modify the large data without copying it. This makes the program use less memory and increases the execution speed.


[memoryview][1]  
[chr][2]   
[operator][3]   
[functools.reduce][4]   
[functools][5]   
[hash][6]   
[bytes][7]   
[ord][8]   
[find][9]   
[reprlib][10]   
[itertools][11]   
[math][12]   
[sys][13]  


[1]:https://www.programiz.com/python-programming/methods/built-in/memoryview/  "memoryview()"  
[2]: https://www.programiz.com/python-programming/methods/built-in/chr/  "chr()"  
[3]: https://docs.python.org/3/library/operator.html/  "operator()"  
[4]: https://www.geeksforgeeks.org/reduce-in-python/   "functools.reduce()"  
[5]: https://docs.python.org/3/library/functools.html/  "functools"  
[6]: https://www.programiz.com/python-programming/methods/built-in/hash/  "hash()"  
[7]: https://www.programiz.com/python-programming/methods/built-in/bytes/  "bytes()"  
[8]: https://www.programiz.com/python-programming/methods/built-in/ord/  "ord()"  
[9]: https://www.w3schools.com/python/ref_string_find.asp/  "find()"  
[10]: https://docs.python.org/3/library/reprlib.html/  "reprlib"  
[11]: https://docs.python.org/3/library/itertools.html  "itertools"  
[12]: https://docs.python.org/3/library/math.html  "math"  
[13]: https://docs.python.org/3/library/sys.html  "sys"  




