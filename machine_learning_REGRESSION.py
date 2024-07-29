import numpy as np

class MachineLearninPrepocessing:
    def __init__(self,x_train):
        self.x_train=x_train
        self.weight=np.random.rand(x_train.shape[1])
        self.bias=np.random.randint(0,100)

    def train_test_split(self, x, y, test_size):
        x_train = x[:int(len(x) * (1 - test_size))]
        x_test = x[int(len(x) * (1 - test_size)):]
        y_train = y[:int(len(y) * (1 - test_size))]
        y_test = y[int(len(y) * (1 - test_size)):]
        
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    def show(self):
        print(self.weight,"w")
        print(self.weight.shape,"W-shape")
        print(self.bias)
        
    def scaling_features(self, data, option="Standard"):
        data = np.array(data, dtype=float)  
        scaled_data = []
        if option == "Standard":
            mean = np.mean(data)
            std = np.std(data)
            for i in data:
                scaled = (i - mean) / std
                scaled_data.append(scaled)
        else:
            minimum = np.min(data)
            maximum = np.max(data)
            for i in data:
                scaled = (i - minimum) / (maximum - minimum)
                scaled_data.append(scaled)
        return np.array(scaled_data)


    
    def hypothesis(self,X):
        return np.array(np.dot(X,self.weight)+self.bias)
    
    def cost_function(self,x_train,y_train):
        predictions=self.hypothesis(x_train)
        cost = 1 / len(x_train) * np.sum(np.square(predictions - y_train))
        return cost
    
    def compute_gradient(self,x_train,y_train):
        m=len(y_train)
        predictions=self.hypothesis(x_train)
        error=predictions-y_train

        dw=(1/m) *np.dot(x_train.T,error)
        db=(1/m) *np.sum(error)

        return dw,db
    
    def gradient_descent(self,lr,num_iter,x_train,y_train):
        costs=[]
        for i in range(0,num_iter):
            dw,db=self.compute_gradient(x_train,y_train)
            self.weight -= lr * dw
            self.bias -= lr * db
            cost=self.cost_function(x_train,y_train)
            costs.append(cost)
            if i%100==0:
                print(f'Iteration {i}, cost: {cost}')
        return costs,self.weight,self.bias
    
    def predict(self,x_test):
        x_test=self.scaling_features(x_test)
        predictions=self.hypothesis(x_test)
        return predictions
    
    def r2_score(self,y_true,y_pred):
        ss_total=np.sum((y_true-np.mean(y_true))**2)
        ss_residual=np.sum((y_true-y_pred)**2)
        r2=1-(ss_residual /ss_total)
        return r2