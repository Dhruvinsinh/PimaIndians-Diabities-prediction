#pima indians  
import numpy as np 
 
inp=[] out=[] with open("/content/pima-indiansdiabetes.csv", 'r') as csvfile:      # creating a csv reader object  
    csvreader = csv.reader(csvfile)  
     
    # extracting field names through first row      for i in csvreader:       temp=[]       temp.append(float(i[0]))       temp.append(float(i[2]))       temp.append(float(i[3]))       temp.append(float(i[4])) 
 
      temp.append(float(i[5]))       temp.append(float(i[6]))       temp.append(float(i[7]))       inp.append(temp)       temp=[]       k=int(i[8])       temp.append(k)       out.append(temp) inp=np.matrix(inp) out=np.matrix(out) 
 
#To solve class imbalance problem from imblearn.over_sampling import SMOTE  sm = SMOTE(random_state = 2)  inp, out = sm.fit_sample(inp,out)  out=out.reshape(-1,1) 
 
#To normalize the data 
from sklearn.preprocessing import Normalizer transformer = Normalizer().fit(inp) inp=transformer.transform(inp) 
new_out=[] for i in out:   k=i[0]   temp=[0,0]   temp[k]=1   new_out.append(temp) out=np.matrix(new_out) import tensorflow as tf k=tf.keras.optimizers.RMSprop(learning_rate=
0.009) model=tf.keras.models.Sequential([tf.keras.laye rs.Input(shape=(7,)),tf.keras.layers.Dense(10,ac tivation='tanh'),tf.keras.layers.Dense(14,activati on='tanh'),tf.keras.layers.Dense(14,activation='t
anh'),tf.keras.layers.Dense(2,activation='sigmoi d')]) 
model.compile(loss='logcosh',optimizer=k,metr ics=['accuracy']) 
model.fit(inp,out,epochs=100) 
