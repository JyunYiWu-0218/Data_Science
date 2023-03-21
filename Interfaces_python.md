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






