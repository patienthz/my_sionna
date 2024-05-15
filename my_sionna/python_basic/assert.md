`assert`语句用于在代码中进行断言，即在特定条件下检查某个表达式是否为真，如果表达式为假，则会触发异常。它的基本语法如下：

```python
assert expression, error_message
```

其中：
- `expression` 是需要检查的表达式，如果表达式为假（False），则会触发异常；
- `error_message` 是可选的错误消息，用于在断言失败时显示。

下面是一个简单的示例：

```python
x = 10
assert x > 0, "x must be positive"
print("x is positive")
```

在这个例子中，如果 `x` 的值小于等于 0，断言就会失败，并且会显示错误消息 `"x must be positive"`，程序会停止执行。

另一个示例是对函数参数进行类型检查：

```python
def square(x):
    assert isinstance(x, (int, float)), "x must be an integer or float"
    return x ** 2

result = square(5)
print(result)  # 输出 25

result = square("hello")
# 断言失败，会显示错误消息："x must be an integer or float"
```

在这个例子中，`assert`语句用于确保函数 `square` 的参数 `x` 是整数或浮点数类型，如果不是，就会触发断言失败的异常。