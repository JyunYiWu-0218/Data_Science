#class method 講解
class Shiba:
    pee_length = 10

    def __init__(self, height, weight):
        self.height = height
        self.weight = weight

    @classmethod
    def pee(cls):
        print("pee" + "." * cls.pee_length)



Shiba.pee()
#result: pee..........


black_shiba = Shiba(30, 40)
black_shiba.pee()
#result: pee.........
'''
class method 必須傳入 class 本身作參數：即例子中的 cls ，這個參數的名稱理論上是可以自由命名的，但一般來說我們都會命名為 cls（代表class 縮寫），
就如同 instance method (一般 method)都會以 self 作為傳入 instance 參數的命名一樣。

和 static method 一樣，class method 也可以由 instance call：在例子中 instance black_shiba 一樣可以 call .pee() 。

和 static method 不同，class method 可以藉由 cls class 本身去 access class static members(variable/method)：
在這個例子中，pee_length 即為 class Shiba 的 static variable，所以 class method 可以利用 cls.pee_length 去 access 到 pee_length 。
但若是 static method 的話就無法了，static method 因為沒有 self 和 cls ，根本無法 access 放在外面的 static member，會直接噴 error。
'''
