import torch
from tqdm import tqdm
from ..core.geometry import get_mean_cov


def e_steering(start_primal, direction, G,
               num_steps = 200, step_size = 1,
               use_tqdm = False, mapping = None, vocab_dict = None):
    current_primal = start_primal.clone()
    primals = [current_primal]
    direction = direction / direction.norm()

    if mapping is not None and vocab_dict is not None:
        indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
        indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]

    step_range = tqdm(range(num_steps)) if use_tqdm else range(num_steps)

    for _ in step_range:
        current_primal = current_primal + direction * step_size
        primals.append(current_primal)

        if mapping is not None and vocab_dict is not None:
            prob = current_primal @ G.T
            prob = torch.softmax(prob, dim=-1)
            prob0 = prob[indices0].sum()
            prob1 = prob[indices1].sum()
            ratio = prob1 / (prob0 + prob1 + 1e-10)
            if ratio > 0.9999:
                break

    primals = torch.stack(primals)
    return primals



def compute_A_p(G_top, prob, mean, p, alpha):
    G_p = torch.matmul(G_top, p)         # [K, D] @ [D] -> [K]
    W_G_p = prob * G_p                   # [K] * [K] -> [K]
    term1 = torch.matmul(G_top.T, W_G_p) # [D, K] @ [K] -> [D]
    
    m_p = torch.dot(mean, p)        
    term2 = mean * m_p                
    
    return term1 - term2 + alpha * p

def cg_solve(G_top, prob, mean, b, alpha, max_iter=20, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for _ in range(max_iter):
        Ap = compute_A_p(G_top, prob, mean, p, alpha)
        
        pAp = torch.dot(p, Ap)
        if pAp.abs() < 1e-10:
            break
            
        alpha_step = rsold / pAp
        x = x + alpha_step * p
        r = r - alpha_step * Ap
        
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x


def m_steering(start_primal, direction, G,
               num_steps=200, step_size=2, alpha=1e-3, topk=20000,
               use_tqdm=True, mapping=None, vocab_dict=None,
               cg_max_iter=20):
    
    if topk is not None and topk > G.shape[0]:
        topk = G.shape[0]

    if mapping is not None and vocab_dict is not None:
        indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
        indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]

    current_primal = start_primal.clone()
    primals = [current_primal]
    
    direction = direction / direction.norm()
    step_range = tqdm(range(num_steps)) if use_tqdm else range(num_steps)
    
    for _ in step_range:
        logit = G @ current_primal
        
        if topk is not None:
            topk_vals, topk_idx = torch.topk(logit, topk)
            prob = F.softmax(topk_vals, dim=-1)
            G_top = G[topk_idx]
            mean = prob @ G_top
        else:
            prob = F.softmax(logit, dim=-1)
            G_top = G
            mean = prob @ G

        sol = cg_solve(G_top, prob, mean, direction, alpha, max_iter=cg_max_iter)
        
        sol = sol / sol.norm()
        current_primal = current_primal + sol * step_size
        primals.append(current_primal.clone())

        if mapping is not None and vocab_dict is not None:
            prob_all = current_primal @ G.T
            prob_all = torch.softmax(prob_all, dim=-1)
            prob0 = prob_all[indices0].sum()
            prob1 = prob_all[indices1].sum()
            ratio = prob1 / (prob0 + prob1 + 1e-10)
            if ratio > 0.9999:
                break
    
    primals = torch.stack(primals)
    return primals