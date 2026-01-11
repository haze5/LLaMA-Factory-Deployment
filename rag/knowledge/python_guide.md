# Python 学习指南

## 基础语法
Python 是一种简洁易学的编程语言。它的语法清晰，适合初学者。

### 变量和数据类型
```python
# 变量赋值
name = "张三"
age = 25
is_student = True

# 数据类型转换
num_str = str(123)
str_num = int("456")
```

### 控制流程
```python
# 条件语句
if age >= 18:
    print("成年人")
else:
    print("未成年")

# 循环
for i in range(5):
    print(i)

# while 循环
count = 0
while count < 3:
    print("Hello")
    count += 1
```

## 函数定义
```python
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b
```

## 常用数据结构

### 列表
```python
fruits = ["apple", "banana", "orange"]
fruits.append("grape")
fruits[0] = "pear"
```

### 字典
```python
person = {
    "name": "张三",
    "age": 25,
    "city": "北京"
}
person["age"] = 26
```

## 文件操作
```python
# 读取文件
with open("file.txt", "r") as f:
    content = f.read()

# 写入文件
with open("file.txt", "w") as f:
    f.write("Hello World")
```