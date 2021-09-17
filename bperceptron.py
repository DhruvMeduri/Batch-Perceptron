import matplotlib.animation as animation
import q5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

# this is for 3D decision boundary of parameter1,2,3
fig = plt.figure()
ax = plt.axes(projection='3d')
x_data=[]
y_data=[]
z_data=[]
for i in range(50):
    x_data.append(q5.data_lst[i][1])
for i in range(50):
    y_data.append(q5.data_lst[i][2])
for i in range(50):
    z_data.append(q5.data_lst[i][3])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', label = 'Setosa');
x_data=[]
y_data=[]
z_data=[]
for i in range(50,100):
    x_data.append(q5.data_lst[i][1])
for i in range(50,100):
    y_data.append(q5.data_lst[i][2])
for i in range(50,100):
    z_data.append(q5.data_lst[i][3])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Reds', label = 'Versicolor');
x_data=[]
y_data=[]
z_data=[]
for i in range(100,150):
    x_data.append(q5.data_lst[i][1])
for i in range(100,150):
    y_data.append(q5.data_lst[i][2])
for i in range(100,150):
    z_data.append(q5.data_lst[i][3])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Blues', label = 'Virginica');

def f1(x, y, result):

    return  (result[0][0] - (result[0][1]*x) - (result[0][2]*y) )/result[0][3]

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f1(X, Y, result1)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('parameter 1')
ax.set_ylabel('parameter 2')
ax.set_zlabel('parameter 3')
ax.axes.set_xlim3d(left=0, right=8)
ax.axes.set_ylim3d(bottom=0, top=8)
ax.axes.set_zlim3d(bottom=0, top=8)
ax.legend()
plt.show()

# this is for 3D decision boundary of parameter 2,3,4
fig = plt.figure()
ax = plt.axes(projection='3d')
x_data=[]
y_data=[]
z_data=[]
for i in range(50):
    x_data.append(q5.data_lst[i][2])
for i in range(50):
    y_data.append(q5.data_lst[i][3])
for i in range(50):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', label = 'Setosa');
x_data=[]
y_data=[]
z_data=[]
for i in range(50,100):
    x_data.append(q5.data_lst[i][2])
for i in range(50,100):
    y_data.append(q5.data_lst[i][3])
for i in range(50,100):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Reds', label = 'Versicolor');
x_data=[]
y_data=[]
z_data=[]
for i in range(100,150):
    x_data.append(q5.data_lst[i][2])
for i in range(100,150):
    y_data.append(q5.data_lst[i][3])
for i in range(100,150):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Blues', label = 'Virginica');

def f2(x, y, result):

    return  (result[0][0] - (result[0][2]*x) - (result[0][3]*y) )/result[0][4]

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f2(X, Y, result1)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('parameter 2')
ax.set_ylabel('parameter 3')
ax.set_zlabel('parameter 4')
ax.axes.set_xlim3d(left=0, right=8)
ax.axes.set_ylim3d(bottom=0, top=8)
ax.axes.set_zlim3d(bottom=0, top=8)
ax.legend()
plt.show()

# this is for 3D decision boundary of parameter1,3,4
fig = plt.figure()
ax = plt.axes(projection='3d')
x_data=[]
y_data=[]
z_data=[]
for i in range(50):
    x_data.append(q5.data_lst[i][1])
for i in range(50):
    y_data.append(q5.data_lst[i][3])
for i in range(50):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', label = 'Setosa');
x_data=[]
y_data=[]
z_data=[]
for i in range(50,100):
    x_data.append(q5.data_lst[i][1])
for i in range(50,100):
    y_data.append(q5.data_lst[i][3])
for i in range(50,100):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Reds', label = 'Versicolor');
x_data=[]
y_data=[]
z_data=[]
for i in range(100,150):
    x_data.append(q5.data_lst[i][1])
for i in range(100,150):
    y_data.append(q5.data_lst[i][3])
for i in range(100,150):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Blues', label = 'Virginica');

def f3(x, y, result):

    return  (result[0][0] - (result[0][1]*x) - (result[0][3]*y) )/result[0][4]

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f3(X, Y, result1)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('parameter 1')
ax.set_ylabel('parameter 3')
ax.set_zlabel('parameter 4')
ax.axes.set_xlim3d(left=0, right=8)
ax.axes.set_ylim3d(bottom=0, top=8)
ax.axes.set_zlim3d(bottom=0, top=8)
ax.legend()
plt.show()
# this is for 3D decision boundary of parameter1,2,4
fig = plt.figure()
ax = plt.axes(projection='3d')
x_data=[]
y_data=[]
z_data=[]
for i in range(50):
    x_data.append(q5.data_lst[i][1])
for i in range(50):
    y_data.append(q5.data_lst[i][2])
for i in range(50):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', label = 'Setosa');
x_data=[]
y_data=[]
z_data=[]
for i in range(50,100):
    x_data.append(q5.data_lst[i][1])
for i in range(50,100):
    y_data.append(q5.data_lst[i][2])
for i in range(50,100):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Reds', label = 'Versicolor');
x_data=[]
y_data=[]
z_data=[]
for i in range(100,150):
    x_data.append(q5.data_lst[i][1])
for i in range(100,150):
    y_data.append(q5.data_lst[i][2])
for i in range(100,150):
    z_data.append(q5.data_lst[i][4])
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Blues', label = 'Virginica');

def f4(x, y, result):

    return  (result[0][0] - (result[0][1]*x) - (result[0][2]*y) )/result[0][4]

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f4(X, Y, result1)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('parameter 1')
ax.set_ylabel('parameter 2')
ax.set_zlabel('parameter 4')
ax.axes.set_xlim3d(left=0, right=8)
ax.axes.set_ylim3d(bottom=0, top=8)
ax.axes.set_zlim3d(bottom=0, top=8)
ax.legend()
plt.show()
'''
# this used to plot by changing learning rate
x = []
for i in range(1,151):
    x.append(i)
for j in range(1,11):
    result1 = bperceptron1(ar,j/10)
    y1 = result1[1]ss
    plt.plot(x,y1)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error Trajectory(Iris-setosa)-Learning Rate="+str(j/10))
    #plt.savefig("learning_rate/fig"+str(j)+".png")
    plt.show()
#print(result1)
'''
