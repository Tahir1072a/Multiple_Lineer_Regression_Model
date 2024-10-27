import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_cost_2(x, y, w, b):
    """
    
    :param x: Burada x artık bir matrixtir. İçinde bir sürü satır bulundurur.
    :param y: 
    :param w: Burada w artık tek boyutlu bir dizidir/vektordür.
    :param b: 
    :return: 
    """
    m = x.shape[0]
    total_cost = 0
    
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost_i = (f_wb - y[i]) ** 2
        total_cost += cost_i
    total_cost = total_cost / 2 * m
    return total_cost

# Şimdi ise gradient descent, yardımcı fonksiyonu olan compute_gradient (eğim hesabı) fonksiyonunu yazalım

def compute_gradient_2(x, y, w, b):
    
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i][j] 
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db
# Son olarak gradient descent fonksiyonumuzu da yazalım.

def gradient_descent_2(x, y, w_init, b_init, iterations, alpha ,cost_function, gradient_function):
    
    J_History = []
    p_History = []
    w = w_init
    b = b_init
    
    for i in range(iterations):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i < 100000:
            J_History.append(cost_function(x,y,w,b))
            p_History.append([w,b])
        
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration: {i:4}: Cost {J_History[-1]:0.2e}")
    
    return w, b, J_History, p_History       
        

# Artık modelimizi oluşturduğumuza göre eğitim verileri üzerinde düzenlemeler yapmamız gerekiyor. Onları en temiz şekilde modelimize vermeliyiz.

data_frame = pd.read_csv("jamb_exam_results.csv")
data_frame.drop(columns=["Student_ID"], inplace=True)
data_frame.dropna(axis=0, how='any', inplace=True)
data_frame.replace({
    "Gender": {"Male": 1, "Female": 0},
    "Parent_Education_Level": {"Primary": 0, "Secondary": 1, "Tertiary": 2},
    "Parent_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_To_Learning_Materials": {"Yes": 1, "No": 0},
    "IT_Knowledge": {"Low": 0, "Medium": 1, "High": 2},
    "Socioeconomic_Status": {"Low": 0, "Medium": 1, "High": 2},
    "Extra_Tutorials": {"Yes": 1, "No": 0},
    "School_Type": {"Public": 0, "Private": 1},
    "School_Location": {"Urban": 0, "Rural": 1}
}, inplace=True)


# Şimdi biraz scailing ile verimizi daha iyi hale getirelim.

datas = data_frame
nn = np.array(datas)

X_data = data_frame.loc[:, data_frame.columns != "JAMB_Score"]
Y_data = data_frame["JAMB_Score"]

X_train = np.array(X_data)
Y_train = np.array(Y_data)

scaler = StandardScaler()
x_norm = scaler.fit_transform(X_train)

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

b_init_2 = 0
w_init_2 = np.zeros(x_norm.shape[1])

num_of_iterations = 1000
temp_A = 9.0e-1

w_f, b_f, J_History, P_history = gradient_descent_2(x_norm, Y_train,w_init_2, b_init_2, num_of_iterations, temp_A,compute_cost_2, compute_gradient_2)

print(f"(w,b) değerleri yapay zeka tarafından tespit edilmiştir: ({w_f, b_f})" )

from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, Y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

x_temp = x_norm[0]
f_wb = np.dot(w_f,x_temp) + b_f
print(f"Actual Value: {Y_train[0]}, Predict: {f_wb}")

f = np.dot(w_norm,x_temp) + b_norm
print(f"Actual Value: {Y_train[0]}, Predict: {f}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_History[:50])
ax2.plot(100 + np.arange(len(J_History[10:])), J_History[10:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

y_pred = sgdr.predict(x_norm)  # y_pred dizinizi uygun bir tahmin dizisi ile değiştirin

# Grafik ayarları
fig, ax = plt.subplots(1, 15, figsize=(30, 5), sharey=True)  # Genişliği artırarak görünümü iyileştiriyoruz

# Her özellik için scatter plot
for i in range(15):
    ax[i].scatter(x_norm[:, i], Y_train, label='Target')
    ax[i].scatter(x_norm[:, i], y_pred, color="red", label='Predict')
    ax[i].set_xlabel(f"Feature {i+1}")

# Ortak Y ekseni etiketi ve başlık
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target versus Prediction using z-score normalized model")
plt.show()
