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
![螢幕擷取畫面 2023-03-20 005813](https://user-images.githubusercontent.com/128043244/226192330-22093ee4-74bb-44e8-a357-e19b699e4497.png)


