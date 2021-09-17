import matplotlib.animation as animation
import q5
import numpy as np
import matplotlib.pyplot as plt

def update_weights(wt,correction,learn):#updates weights after every epoch
    wt = wt+ (learn*correction)
    return wt
def compute_error1(wt):#computes #misclassified data points with wt wrt Setosa
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
        if q5.data_lst[i][5]!='Iris-setosa' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-setosa' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err
def compute_error2(wt):#computes #misclassified data points with wt wrt Versicolor
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
        if q5.data_lst[i][5]!='Iris-versicolor' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-versicolor' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err
def compute_error3(wt):#computes #misclassified data points with wt wrt Virginica
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
        if q5.data_lst[i][5]!='Iris-virginica' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-virginica' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err

def bperceptron1 (ar,learn):#This function is for the classification of Iris-Setosa>=0 and !Iris Setosa<0 using batch perceptron
   err_lst=[]
   weights=np.array([1,2,3,4,5])
   #res = list(np.random.randint(low = 1,high=6,size=5))# this is to randomize the weights
   #weights = np.array(res)
   #ar = np.random.permutation(150)
   for j in range(100):
      ar = np.random.permutation(150)
      correction = np.array([0,0,0,0,0])
      for i in ar:
           temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
           if q5.data_lst[i][5] != 'Iris-setosa' and np.dot(weights,temp_vec)>=0:
                correction = correction - temp_vec
           if q5.data_lst[i][5] == 'Iris-setosa' and np.dot(weights,temp_vec)<0:
               correction = correction + temp_vec
      weights = update_weights(weights,correction,learn)
      err_lst.append(compute_error1(weights))
   print(weights)
   return(weights,err_lst)

def bperceptron2 (ar,learn):#This function is for the classification of Iris-Versicolor>=0 and !Iris Versicolor<0 using batch perceptron
   err_lst=[]
   weights=np.array([0,0,0,0,0])
   #res = list(np.random.randint(low = 1,high=6,size=5))#this is to randomize the weights
   #weights = np.array(res)
   #ar = np.random.permutation(150)
   for j in range(100):
      ar = np.random.permutation(150)
      correction = np.array([0,0,0,0,0])
      for i in ar:
           temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
           if q5.data_lst[i][5] != 'Iris-versicolor' and np.dot(weights,temp_vec)>=0:
                correction = correction - temp_vec
           if q5.data_lst[i][5] == 'Iris-versicolor' and weights.dot(temp_vec)<0:
               correction = correction + temp_vec
      weights = update_weights(weights,correction,learn)
      err_lst.append(compute_error1(weights))
   return(weights,err_lst)

def bperceptron3 (ar,learn):#This function is for the classification of Iris-Virginica>=0 and !Iris-Virginica<0 using batch perceptron
   err_lst=[]
   weights=np.array([0,0,0,0,0])
   #res = list(np.random.randint(low = 1,high=6,size=5))#this is to randomize the weights
   #weights = np.array(res)
   #ar = np.random.permutation(150)
   for j in range(100):
      ar = np.random.permutation(150)
      correction = np.array([0,0,0,0,0])
      for i in ar:
           temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
           if q5.data_lst[i][5] != 'Iris-virginica' and np.dot(temp_vec,weights)>=0:
                correction = correction - temp_vec
           if q5.data_lst[i][5] == 'Iris-virginica' and np.dot(temp_vec,weights)<0:
               correction = correction + temp_vec
      weights = update_weights(weights,correction,learn)
      correction=np.array([0,0,0,0,0])
      err_lst.append(compute_error1(weights))
   return(weights,err_lst)


ar=np.random.permutation(150)
result1 = bperceptron1(ar,1)
result2 = bperceptron2(ar,1)
result3 = bperceptron3(ar,1)
x = []
for i in range(1,101):
      x.append(i)
y1 = result1[1]
plt.plot(x,y1)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error Trajectory(Iris-setosa or not)")
#plt.savefig("q5a/fig1")
plt.show()

y2 = result2[1]
plt.plot(x,y2)
plt.xlabel("Epochs")
plt.ylabel("Error ")
plt.title("Error Trajectory(Iris-Versicolor or not)")
#plt.savefig("q5a/fig2")
plt.show()

y3 = result3[1]
plt.plot(x,y3)
plt.xlabel("Epochs")
plt.ylabel("Error ")
plt.title("Error Trajectory(Iris-Virginica or not)")
#plt.savefig("q5a/fig3")
plt.show()
#shuffling was just done by running the same code multiple times, it was shuffling
#because of the random.permutation function. The same above code was ran multiple
#times with random initial weights
'''
#plotting decision boundary on parameter 1 and 2
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][2]) - (result1[0][1]/result1[0][2])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter2")
plt.legend()
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig4")
plt.show()
#plotting decision boundary on parameter 1 and 3
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][3]) - (result1[0][1]/result1[0][3])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter3")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig5")
plt.show()

#plotting decision boundary on parameter 1 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][1]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig6")
plt.show()

#plotting decision boundary on parameter 2 and 3
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][3]) - (result1[0][2]/result1[0][3])*x
plt.plot(x,y)
plt.xlabel("Parameter2")
plt.ylabel("Parameter3")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig7")
plt.show()

#plotting decision boundary on parameter 2 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][2]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter2")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig8")

plt.show()


#plotting decision boundary on parameter 3 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][3]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter3")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/Fig9")
plt.show()
'''
'''
#This is plotting varying learning rate
for i in range(1,11):
    result1 = bperceptron1(ar,i/10)
    y1 = result1[1]
    plt.plot(x,y1)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error Trajectory(Iris-setosa or not)-Learning Rate:"+str(i/10))
    plt.savefig("learning_rate/fig"+str(i/10)+".png")
    plt.show()
'''
