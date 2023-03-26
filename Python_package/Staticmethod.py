#Static Method
'''
什麼是 static ?!
以 Java 為例（對於大多數語言static的核心概念是相似的），
一個 member 被宣告為 static 代表該 member 是在 Class Initialization 階段就被寫在一塊記憶體上，
因為是發生在 Instance Initialization 前，所以理所當然不會帶有 instance，且該 static member 之參數必須也是 static的（不然根本拿不到）。
而該 class 所生之所有 instances 皆可以使用該 static member 。

總結，使用 static 的時機有：

1.Design 意義上，希望某個 member independent of instance，不帶instance為參數，就會宣告該 member 為 static ，使 members 間的關係更加乾淨俐落。

2.Design 意義上，希望某個 variable 是 static 的代表，想要該 class 所生之所有 instances 共用相同值的參數，作為 instances 間的溝通與合作。

3.Physical 意義上，一個 static member 因為是只使用到一塊記憶體，故 instances 無論多寡，該 static member 所佔空間都不會增加。
'''
class Shiba:
    def __init__(self, height, weight):
        self.height = height
        self.weight = weight

    @staticmethod
    def pee(length):
        print("pee" + "." * length)


Shiba.pee(3)
# Result > pee...

Shiba.pee(20)
# Result > pee....................

black_shiba = Shiba(90, 40)
black_shiba.pee(10)
# Result > pee...........
'''
1.static method 不需要也不能將 instance 以參數 self 的方式傳入

2.static method 可以由 class 直接 call 而不一定需要用到 instance call

3.static method 也可以由 instance (例子中的 black_shiba) call，但是是不帶有 instance參數的。
'''
