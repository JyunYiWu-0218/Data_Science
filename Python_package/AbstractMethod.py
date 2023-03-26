#Abstract Method
"""
1. 什麼是 abstract method ?!
來到要介紹的最後一種 method，abstract method。
首先澄清一下，如果熟悉 Object Oriented 概念的人一定知道什麼是 polymorphism (多形)的概念，
在 Java 上簡單來說就是以 interface 實現，而在 C++ 上是以 pure virtual function 實現。

Polymorphism 是 OOP 中重要的 design ，有以下幾個最重要的優點：

1.增加 programming 的維護性：
使用 polymorphism 相當於把該 object 的接口都定義好，
讓其他 object 在 inherit 或是 implement 時有一個「代辦清單」，而且是一個「定義嚴謹的代辦清單」（裡頭包含 input/output 的 type 等等）。

2.減低 object 彼此的依賴度：
舉例來說，原本 A 、B 和 C 三個 objects (classes) 都要實作 吃大便() 和 喝飲料() 這兩個 method，
為了展現 A 、B 和 C是有關係的，所以會採用 inheritance 方式去實作，即 B inherits A，C inherits B，某天當 A 要做改變時，
就會連帶影響 B 和 C。如果今天採用polymorphism 的方式設計，相當於「把結構從鏈狀、上下游的裙帶關係」轉成「只有一個控制中心而底下為平行關係」

實現 abstract method 有兩種方式：
一種較為寬鬆（只在 call method 時才去檢查 abstract method 是否已經被 implemented）。

另一種較為嚴謹正規（在 class initialization 時即會檢查 abstract method 是否已經被實作）。


如何建立 abstract methods 呢？
先將 class 的 metaclass 設成 abc.ABCMeta ，並且直接在 method 上添加 @abc.abstractmethod 就可以將該 method 變成 abstract 了！
（當我們在某個 method 上加 @abc.abstractmethod 時，預設為 abstract instance method 唷！）
當我們在最後一行 code 中 initialize Shiba 這個 class 時，就會因為我們沒有 override abstract methods （去實現這個 abstract method）而噴 error。
"""
import abc


class Dog(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eat_shit(self):
        return NotImplemented

    @abc.abstractmethod
    def pee(self):
        return NotImplemented


class Shiba(Dog):
    def eat_shit(self):
        print("I'm eating shit".format())

    def pee(self):
        print("I'm peeing........")


Shiba().eat_shit()
# Result > I'm eating shit

Shiba().pee()
# Result > I'm peeing........
"""
將 Dog 作為 interface ，讓 Shiba 以 inherit 的方式並且 override eat_shit() 和 pee() 這兩個 methods，這樣就達到實作 interface 的效果了！！！

interface(介面)：
介面的重點：

介面中的資料成員必須設定初值
介面沒有一般函數，只有抽象函數

inherit(繼承):
基於某個父類別對物件的定義加以擴充，而制訂出一個新的子類別定義，子類別可以繼承父類別原來的某些定義，
並也可能增加原來的父類別所沒有的定義，或者是重新定義父類別中的某些特性。

"""
