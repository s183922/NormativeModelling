import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt 
import nilearn
import nibabel
from nilearn import plotting
from nilearn import datasets
import os, sys; os.chdir(sys.path[0])
from sklearn import model_selection
from AUC_load import load_input_output, slicer
from tqdm import tqdm
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_dims = 3, batch_shape = None):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.x = train_x
        self.y = train_y
        
        

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean(batch_shape = batch_shape)
        # self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module_Linear = gpytorch.kernels.LinearKernel(batch_shape = batch_shape)
        self.covar_module_SE = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = num_dims,
         batch_shape=batch_shape), batch_shape = batch_shape)

        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(self.covar_module_Linear + self.covar_module_SE, num_dims)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x,  covar_x)
        

    def fit_hp(self, train_iter = 100):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.1)
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
    
        
        for i in range(train_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.forward(self.x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.y).sum()
            loss.backward()
            # print(f'Iter {i+1} - Loss: {loss.item()}')
            print(f'Iter {i+1} - Loss: {round(loss.cpu().item(),4)}, mean lengthscale: {[round(a,4) for a in self.covar_module_SE.base_kernel.lengthscale.mean(0).cpu().detach().numpy()[0]]},   mean noise: {round(self.likelihood.noise.mean().cpu().item(),4)}, mean SE_variance: {round(self.covar_module_SE.outputscale.mean().cpu().detach().item(), 4)}, mean_linear_variance: {round(self.covar_module_Linear.variance.mean().cpu().detach().item(),4)}')
            optimizer.step()
         
class prior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_dims = 3  ):
        super(prior, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module_Linear = gpytorch.kernels.LinearKernel()
        self.covar_module_SE = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = 3))

        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(self.covar_module_Linear + self.covar_module_SE, 3)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x,  covar_x)


def Cross_validation(X, y, labels, Kfold = 10, n_iter = 100, step = 5000):
    CV = model_selection.KFold(n_splits = Kfold, shuffle = True, random_state = 2020)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (train_index, test_index) in enumerate(tqdm(CV.split(X[1]))):
        if i > 4:
            X_train, y_train = X[:,train_index,:].to(device), y[:,train_index].to(device)
            X_test = X[:,test_index,:].to(device)
            mean = []
            var  = []
            for S in tqdm(slicer(y_train, step)):
                X_train_S, y_train_S = X_train[S], y_train[S]
                X_test_S = X_test[S]

                likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape = torch.Size([X_train_S.shape[0]])).to(device)
                model = GPModel(X_train_S, y_train_S, likelihood, batch_shape = torch.Size([X_train_S.shape[0]])).to(device)
                model.fit_hp(train_iter=n_iter)


                with torch.no_grad():

                    model.eval()
                    likelihood.eval()

                    prediction = likelihood(model(X_test_S))
                    mean.append(prediction.mean.detach())
                    var.append(prediction.variance.detach())
            mean = torch.cat(mean)
            var = torch.cat(var)
            [torch.save(mean[:,i].cpu(), "./Data/pred_mean/" + str(name) + "_mu" + ".pt") for i, name in enumerate(labels[test_index])]
            [torch.save(var[:,i].cpu(), "./Data/pred_var/" +str(name) + "_var" + ".pt") for i, name in enumerate(labels[test_index])]







if __name__ == '__main__':
    # with torch.no_grad():
    #     X, y, labels = load_input_output()
    #     y = (y-y.mean(0))/y.std(0)
    # Cross_validation(X,y, labels, Kfold = 10, n_iter = 100, step = 500)
    X_train = torch.linspace(60,90,100)
    y_train = torch.randn(1,100)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = prior(X_train, y_train, likelihood, num_dims= 1)
    x_test = torch.linspace(-10,10,100)
    model.eval()
    likelihood.eval()
    with gpytorch.settings.prior_mode(True):
        SE_output = 0.07
        L_output = 1.1e-5
        lengthscale = 2.3
        model.covar_module_SE.outputscale = torch.Tensor([SE_output])
        model.covar_module_Linear.variance = torch.Tensor([L_output])
        model.covar_module_SE.base_kernel.lengthscale = torch.Tensor([lengthscale])
        


        pred_linear = gpytorch.distributions.MultivariateNormal(model.mean_module(x_test),  model.covar_module_Linear(x_test))
        pred_SE = gpytorch.distributions.MultivariateNormal(model.mean_module(x_test),  model.covar_module_SE(x_test))
        pred_add =likelihood( pred_SE + pred_linear)
        fig = plt.figure(figsize=(20,10))
        
        X = x_test.detach().numpy()
        for i in range(1):
            l = pred_linear.rsample().detach().numpy()
            se = pred_SE.rsample().detach().numpy()
            # a = l + se 
            


            fig.add_subplot(1,3,1)
            plt.plot(X,l)
            plt.xlim(-10,10)
            plt.ylim(-2,2)
            plt.title("Prior - Linear Covariance",fontsize=20)

            
            fig.add_subplot(1,3,2)
            plt.plot(X,se)
            plt.fill_between(X, lower.detach().numpy(), upper.detach().numpy())
            plt.xlim(-10,10)
            plt.ylim(-5,5)
            plt.title("Prior - SE Covariance",fontsize=20)


            lower, upper = pred_add.confidence_region()
            fig.add_subplot(1,3,3)
            plt.plot(X,pred_add.mean.detach().numpy())
            plt.fill_between(X, lower.detach().numpy(), upper.detach().numpy())
            plt.xlim(-10,10)
            plt.ylim(-5,5)
            plt.title("Prior - Linear + SE Covariance",fontsize=20)
           

        plt.show()


        # for i in range(20):
        #     plt.plot(x_test.detach().numpy(),pred_linear.rsample().detach().numpy())
        #     plt.xlim(-10,10)
            
        #     plt.ylim(-3,3)
        #     plt.title("Prior - Linear Covariance",fontsize=20)
        # fig.add_subplot(1,3,2) 
        # for i in range(20):
        #     plt.plot(x_test.detach().numpy(),pred_SE.rsample().detach().numpy())
        #     plt.xlim(-10,10)
        #     plt.ylim(-3,3)
        #     plt.title("Prior - SE Covariance", fontsize=20)
        # fig.add_subplot(1,3,3) 
        # for i in range(20):
        #     plt.plot(x_test.detach().numpy(),pred_add.rsample().detach().numpy())
        #     plt.xlim(-10,10)
        #     plt.ylim(-3,3)
        #     plt.title("Prior - Linear + SE Covariance",fontsize=20)
        # plt.show()   



