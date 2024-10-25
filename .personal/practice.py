import matplotlib.pyplot as plt
a = [1,2,3,4,5,6,7]
b = [11,22,33,44,55,66,77]
plt.plot(a, b)
plt.show()




print(np.sum(a, axis = 1))
print(a.T)
print(np.linalg.det(a))




#b = np.array ([[3,2,1],[6,5,4],[9,8,7]])
#d = np.dot(a,b)
#print(d)




a = np.array ([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(np.sum(a[2, :]))