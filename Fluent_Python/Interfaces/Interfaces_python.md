# Interfaces: From Protocols to ABCs   
***An abstract class represents an interface.***    
**Don’t want to encourage you to start writing your own ABCs left and right. The risk of over-engineering with ABCs is very high.**    
ABCs, like descriptors and metaclasses, are tools for building frameworks.    
---

## Interfaces and Protocols in Python  
Protocols are defined as the informal interfaces that make polymorphism(多型) work in languages with dynamic typing like Python.  
  
***How do interfaces work in a dynamic-typed language?***  
the basics: ~~even without an interface keyword in the language, and regardless of ABCs~~  
every class has an interface: the set public attributes (methods or data attributes) implemented or inherited by the class.  
protected and private attributes are not part of an interface, even if “protected” is merely a naming convention (the single leading underscore).  
it’s not a sin to have public data attributes as part of the interface of an object, because a data attribute can always be turned into a property implementing getter/setter logic without breaking client code that uses the plain
obj.attr syntax(語法).  
***complementary definition of interface is: the subset of an object’s public methds that enable it to play a specific role in the system.***  
That’s what is implied when the Python documentation mentions “a file-like object” or “an iterable,” without specifying a class.  
***Protocols are independent of inheritance, Protocols are interfaces, but because they are informal—defined only by documentation(文檔) and conventions—protocols cannot be enforced like formal interfaces can.***        
A class may implement several protocols, enabling its instances to fulfill several roles.  
  
  
## Python Sequences
***The philosophy of the Python data model is to cooperate with essential protocols as much as possible.***  
When it comes to sequences, Python tries hard to work with even the simplest implementations.  
![Sequence_ABC_UML](https://user-images.githubusercontent.com/128043244/226349499-7d6605c2-cf8f-4615-8416-8ea9a812f5c7.png "Sequence_ABC_UML")
![__getitem__](https://user-images.githubusercontent.com/128043244/226350253-a69a2f13-c01e-4f1e-8342-c5e6dda9738f.png "Partial sequence protocol implementation with __getitem__")
No method __ iter __ yet Foo instances are iterable because—as a fallback—when Python sees a __ getitem __ method, it tries to iterate over the object by calling that method with integer indexes starting with 0.      
It can also make the in operator work even if Foo has no _ _contains_ _ method: it does a full scan to check if an item is present.      
given the importance of the sequence protocol, in the absence  __ iter __ and __ contains __ Python still manages to make iteration and the in operator work by
invoking __ getitem __.     
ex: French.py (methods of the sequence protocol: __ getitem __ and __ len __.)    
Iteration in Python represents an extreme form of duck typing:(the interpreter tries two different methods to iterate over objects.)  

## Monkey-Patching at Runtime  
![Monkey_Patching](https://user-images.githubusercontent.com/128043244/226509758-fd37becf-fa83-454b-9c81-9e110a6baafd.png  "Monkey-Patching")
The problem is that shuffle operates by swapping items inside the collection, and FrenchDeck only implements the immutable sequence protocol.   
Mutable sequences must also provide a __ setitem __ method.  
***Monkey patching FrenchDeck to make it mutable and compatible with random.shuffle:***      
![mutable_and_____compatible](https://user-images.githubusercontent.com/128043244/226512393-1e4306e5-ea77-4bb8-827b-ed5d000adbfe.png)
![__setitem__method](https://www.geeksforgeeks.org/__getitem__-and-__setitem__-in-python/):    
object.setitem(self, key, value), Called to implement assignment to self[key].Same note as for getitem().      
This should only be implemented for mappings if the objects support changes to the values for keys, or if new keys can be added, or for sequences if elements can be replaced. The same exceptions should be raised for improper key values as for the getitem() method.  
Monkey Patching is powerful, but the code that does the actual patching is very tightly(實際地) coupled with the program to be patched, often handling private and undocumented parts.  
  
## Duck Typing commonly used in Python :  
***Duck Typing : (A style of dynamic typing)***    
In this style, the effective semantics of an object is not determined by inheriting from a specific class or implementing a specific interface, but by "the current set of methods and properties".  
**In other words, the focus is on the behavior of the object, what it can do; not on the type of the object.**  

## Subclassing an Collections_ABC
leverage an existing ABC, collections.MutableSequence, before daring to invent our own.
![Frenchdeck2.py](https://github.com/JyunYiWu-0218/Data_Science/blob/Python/Fluent_Python/Interfaces/Frenchdeck2.py)  
![Double_Underscore_Method(magic_method)](https://blog.finxter.com/python-list-of-dunder-methods/)  
Python does not check for the implementation of the abstract methods at import time(when the frenchdeck2.py module is loaded and compiled), but only at runtime when
we actually try to instantiate FrenchDeck2.  
If fail to implement any abstract method, get a TypeError exception with a message such as "Can't instantiate abstract class FrenchDeck2 with abstract methods 
delitem method, insert".    
That’s why implement delitem method and insert, even if our FrenchDeck2 examples do not need those behaviors: the MutableSequence ABC demands them.  
![MutableSequence_ABCs](https://user-images.githubusercontent.com/128043244/228121863-e5975a58-8df7-40a1-a11f-2d5a9b6a1265.png)  








