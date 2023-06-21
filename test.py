class foo:
    def __init__(self):
        self.mypre = [5, 8]

f = foo()
end = f.mypre[1]
end = end + 5
print(f.mypre)