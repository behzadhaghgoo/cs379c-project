import numpy as np
import torch

from .utils import Variable


class TDLoss():

    def __init__(self,batch_size = 32, theta = 1, mode = "dot", exp = False, meg_norm = False, method = 'PER'):
        """Args:
         mode: "dot" or "euc", the distance function for averaging
         theta: power of weights (see paper) """
        super(TDLoss, self).__init__()
        self.batch_size = batch_size
        self.theta = theta
        self.mode = mode
        self.exp = exp
        self.meg_norm = meg_norm
        self.hidden = "hidden"
        self.method = method 
        self.gamma = 0.99

    def hidden_weights(self, h):
        # TODO: rename normalized norm for clarity, implement normalized and unnormalized version (unnormalized
        # seems to be standard in CS224n coverage of attention)
        if self.meg_norm:
          # Meg Norm: w[i,j] = h[i].h[j]/|h[i]||h[j]|
          h = h / torch.reshape(torch.norm(torch.mul(h, h), dim = 1, p = 2), (-1,1))
        
        weights = torch.mm(h,torch.transpose(h, 0, 1))
        # Make weights positive and normalize with softmax (maps (-inf, inf) to [0,1] while maintaining ordering)
        weights = torch.nn.functional.softmax(weights, dim=1)**self.theta
        return weights

    def hidden_mean(self, h, tensor):
        if self.exp:
            tensor = torch.exp(tensor)
        tensor = torch.reshape(tensor,(-1,1))

        if self.mode == "dot":
            weights = self.hidden_weights(h)
        elif self.mode == "euc":
            # TODO: Implement euclidean_weights
            raise Exception("euclidean_weights not implemented")

        output = torch.mm(weights, tensor)
        output = output.squeeze(1)
        if self.exp:
            return torch.log(output * self.batch_size)
        return output * self.batch_size

    def compute_td_loss(self, cur_model, tar_model, beta, replay_buffer, optimizer):
        state, action, reward, next_state, done, indices, weights, state_envs = replay_buffer.sample(self.batch_size, beta)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))
        
        # predict q value and store hidden state if averaging q values
        if self.method == 'average_over_batch':#average_q_values:
            q_values, hiddens = cur_model.forward(state, return_latent = "last")
        else:
            q_values, hiddens = cur_model.forward(state, return_latent = None)
        next_q_values, _ = tar_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        loss  = loss.mean()

        if self.method == 'average_over_batch': #average_q_values:
            # TODO: computing average over only sampled hidden states is limiting. Ideal case: compute over
            # entire buffer each time. More realistic, have buffer that stores hidden state (limits size of buffer,
            # risk of stale entries if large). Interesting spot for experimentation
            q_value = self.hidden_mean(hiddens, q_value)
            # should not be averaging expected q value--the goal of the algorithm is to have each individual
            # prediction be close to the averaged prediction
            # possible issue with task as it is defined now: it is beneficial to average noisy tasks with non-noisy
            # (though not the reverse)
            # TODO: investigate hidden state similarity
#              expected_q_value = self.hidden_mean(hiddens, expected_q_value)

        prios = (q_value - expected_q_value.detach()).pow(2) * weights + 1e-5

        optimizer.zero_grad()
        loss.backward()
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        optimizer.step()

        return loss
