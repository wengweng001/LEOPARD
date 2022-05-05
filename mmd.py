import torch
import torch.nn as nn

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def mmd_g(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

    
def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))
    return mmd2, varEst, Kxyxy


# def MMDu(Fea1, Fea2, Fea_org1, Fea_org2, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
#     """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
#     X = Fea1 # fetch the sample 1 (features of deep networks)
#     Y = Fea2 # fetch the sample 2 (features of deep networks)
#     X_org = Fea_org1 # fetch the original sample 1
#     Y_org = Fea_org2 # fetch the original sample 2
#     L = 1 # generalized Gaussian (if L>1)
#     Dxx = Pdist2(X, X)
#     Dyy = Pdist2(Y, Y)
#     Dxy = Pdist2(X, Y)
#     Dxx_org = Pdist2(X_org, X_org)
#     Dyy_org = Pdist2(Y_org, Y_org)
#     Dxy_org = Pdist2(X_org, Y_org)
#     if is_smooth:
#         Kx = (1.0-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma)) + epsilon * torch.exp(-Dxx_org / sigma)
#         Ky = (1.0-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma)) + epsilon * torch.exp(-Dyy_org / sigma)
#         Kxy = (1.0-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma)) + epsilon * torch.exp(-Dxy_org / sigma)
#     else:
#         Kx = torch.exp(-Dxx / sigma0)
#         Ky = torch.exp(-Dyy / sigma0)
#         Kxy = torch.exp(-Dxy / sigma0)
#     return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)
    
def MMDu_multi(weights, Fea1, Fea2, Fea_org1, Fea_org2, sigma, sigma0, epsilon, is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    num_kernels = len(Fea1)
    X_org = Fea_org1 # fetch the original sample 1
    Y_org = Fea_org2 # fetch the original sample 2
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    for ii in range(num_kernels):
        X = Fea1[ii]  # fetch the sample 1 (features of deep networks)
        Y = Fea2[ii]  # fetch the sample 2 (features of deep networks)
        Dxx = Pdist2(X, X)
        Dyy = Pdist2(Y, Y)
        Dxy = Pdist2(X, Y)

        if is_smooth:
            Kx = (1.0-epsilon[ii]) * torch.exp(-(Dxx / sigma0[ii]) - (Dxx_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxx_org / sigma[ii])
            Ky = (1.0-epsilon[ii]) * torch.exp(-(Dyy / sigma0[ii]) - (Dyy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dyy_org / sigma[ii])
            Kxy = (1.0-epsilon[ii]) * torch.exp(-(Dxy / sigma0[ii]) - (Dxy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxy_org / sigma[ii])
        else:
            Kx = torch.exp(-Dxx / sigma0[ii])
            Ky = torch.exp(-Dyy / sigma0[ii])
            Kxy = torch.exp(-Dxy / sigma0[ii])

        if ii == 0:
            Kx_all = weights[ii] * Kx
            Ky_all = weights[ii] * Ky
            Kxy_all = weights[ii] * Kxy
        else:
            Kx_all = Kx_all + weights[ii] * Kx
            Ky_all = Ky_all + weights[ii] * Ky
            Kxy_all = Kxy_all + weights[ii] * Kxy

    return h1_mean_var_gram(Kx_all, Ky_all, Kxy_all, is_var_computed, use_1sample_U)

def MMDu_linear_kernel(Fea1, Fea2, is_var_computed=True, use_1sample_U=True):
    """compute value of (deep) lineaer-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
    try:
        X = Fea1
        Y = Fea2
    except:
        X = Fea1.unsqueeze(1)
        Y = Fea2.unsqueeze(1)
    Kx = X.mm(X.transpose(0,1))
    Ky = Y.mm(Y.transpose(0,1))
    Kxy = X.mm(Y.transpose(0,1))
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

import numpy as np

is_cuda = True

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x
    
def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, dtype, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()


def TST_MMD_adaptive_bandwidth(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, is_smooth=False)
    mmd_value = get_item(TEMP[0],is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]
        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()
