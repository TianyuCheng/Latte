    def forward(self):
        # innder product of inputs and weights
        assert len(self.forward_adj) > 0, "No forward adjacency element. "
        self.output = 0.0 
       #------------------------------------------------------
        for prev in self.backward_adj:
            pi = prev.pos_x
            pj = prev.pos_y
            self.output += self.weights[pi][pj] * self.inputs[pi][pj]
       #------------------------------------------------------
       # activation
        self.output = np.tanh(dp_result) 
        # preset the gradient for back propagation
        self.grad_activation = (1 - np.tanh(dp_result) ** 2 )

    def backward(self):
        self.grad_output = self.grad_output * self.grad_activation
        # backpropagate error
        for prev in self.backward_adj:
            pi = prev.pos_x
            pj = prev.pos_y
            prev.grad_output += self.grad_output * self.weights[pi][pj]
        # weights to update
        for prev in self.backward_adj:
            pi = prev.pos_x
            pj = prev.pos_y
            self.grad_weights[pi][pj] += self.grad_output * self.inputs[pi][pj]


# backward() prev.grad_output += self.grad_output * self.weights[pi][pj]
for (int i = ; i < ; i )  {

    for (int j = ; j < ; j ) {
        CUR_ENM_grad_output[i][j] += self.grad_output[x][y] * (self.weights[x][y] + i*___ + j)

    }

}
# backward() self.grad_weights[pi][pj] += self.grad_output * self.inputs[pi][pj]
for (int i = ; i < ; i )  {

    for (int j = ; j < ; j ) {
        CUR_ENM_grad_weights[i][j] += CUR_ENM_grad_output[x][y] * (CUR_ENM_weights[x][y] + i*___ + j)

    }

}

# forward() self.output += self.weights[pi][pj] * self.inputs[pi][pj]
for (int i = 0; i < ; i) {
    
    for (int j = ; j < ; j ) {
        CUR_ENM_OUTPUT += *(self.weights[x][y] +i*___ + j) * (*(PREV_ENM_output+i*___+j))

    }
}

