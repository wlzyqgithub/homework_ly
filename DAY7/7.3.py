file = open('test.txt','r+')
old = file.read()
file.close()
file = open('test.txt','w')
file.write('python')
file.write(old)
file.write('python')