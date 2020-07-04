import torch
from torch import nn
from torch import optim


class fcNet(nn.Module):
    '''Fully-connected (FC) network model.
    Output layer has no activation, i.e. it's a plain linear layer.
    '''
    def __init__(self,input_dim=1,layer_dims=[],output_dim=1, dropout_p=[]):
        '''
        Arg-s:
        - input_dim : input dimensions, e.g. 3
        - layer_dims : list of dimensions for hidden layers (layer sizes)
        - output_dim: output dimensions {default: 1}
        - dropout_p : a list of dropout probabilities.
        For each layer if len==len(layer_dims), and dropout prob-y 
        for all layers if len==1, and no dropout if len==0. 
        
        input_layer-->hidden_layers-->output_layer
        
        '''
        super(fcNet,self).__init__()
        assert len(layer_dims)>0
        assert len(dropout_p)==0 or len(dropout_p)==1 or len(dropout_p)==len(layer_dims)

        if len(dropout_p)==1:
            dropout_p = [dropout_p[0] for k in range(len(layer_dims)) ]
        
        hidden_layers = [('linear1',nn.Linear(input_dim, layer_dims[0])),
                        ('activation1',nn.ReLU()) ]# input->layer1

        if len(dropout_p)>0:
            if dropout_p[0]>0:
                hidden_layers.append(('dropout1',nn.Dropout(p= dropout_p[0])))

        for k in range(len(layer_dims)-1):
            hidden_layers.append(
                ('linear'+str(k+2), nn.Linear(layer_dims[k], layer_dims[k+1]) ) )
            hidden_layers.append(
                ('activation'+str(k+2), nn.ReLU() ) )
            if len(dropout_p)>0:
                if dropout_p[k+1]>0:
                    hidden_layers.append(
                            ('dropout'+str(k+2),nn.Dropout(p= dropout_p[k+1]) ) )

        
        self.hidden_layers = nn.ModuleDict(hidden_layers)
        self.output_layer  = nn.Linear(layer_dims[-1], output_dim)

    def forward(self, x):
        '''x: input vector'''
        for l in self.hidden_layers:
            x = self.hidden_layers[l](x)
        return self.output_layer(x)


class diffNet(nn.Module):
    '''
    Difference network. Computes y(T+lead_time) = diffNet(x, y(T+0)).
    
    diffNet(x, y(T+0)) = y(T+0) + fcNet(x), where fcNet is the fully-connected (FC) neural net.
    '''
    def __init__(self,**kwargs):
        '''
        Arg-s for FC part are same as `fcNet`'s:
        - input_dim : input dimensions, e.g. 3
        - layer_dims : list of dimensions for hidden layers (layer sizes)
        - output_dim: output dimensions {default: 1}
        - dropout_p : a list of dropout probabilities.
        For each layer if len==len(layer_dims), and dropout prob-y 
        for all layers if len==1, and no dropout if len==0. 

        Model:
        input_layer=x -->hidden_layers--> output_layer=dY
        Y = Y0 + dY(x)
        '''
        super(diffNet,self).__init__()

        self.net = fcNet(**kwargs)
        
    def forward(self, x, Y0):
        return self.net(x) + Y0



def get_model(net_type='fcNet', opt_type = 'Adam',
              model_kwargs = {}, solver_kwargs = {}, device=torch.device("cpu")):
    '''
    Arg-s:
    - net_type : type of the network model, one of ['fcNet', 'diffNet'].
    - opt_type : type of solver/optimizer one of ['Adam', 'SGD'].
    - model_kwargs : dict of arguments for the model,
    e.g. {input_dim=5, layer_dims=[8,4], output_dim=1}
    - solver_kwargs: dict of arg-s for solver/optimizer
    '''
    models = {'fcNet': fcNet, 'diffNet':diffNet}
    model = models[net_type](**model_kwargs).to(device)
    
    optims = {'Adam': optim.Adam, 'SGD': optim.SGD}
    opt = optims[opt_type](model.parameters(), **solver_kwargs)
    return model, opt


def train_model(model,x,y,ValData,optim, loss_func,
                epochs=100, test_intervl=25, print_times=10, scheduler=None):
    '''
    Arg-s:
    - x : input vectors (a tensor): x , or tuple of tensors :(x, Y0) for difference nets.
    - y : targets/labels ("actual y")
    - ValData : tuple (x_val,y_val), validation or testing dataset,
    for diffnet tuple of tuples ((x,Y0),y)
    - optim : optimizer/ solver
    - test_intervl: collect loss every "test_intervl" iter-n
    - print_times : number of times to print loss to the terminal
    - epochs : total number of training epochs == training iterations for the batch
    '''
    x_val, y_val = ValData
    is_diff_net_ = False
    if isinstance(x, tuple):
        is_diff_net_=True
        x, Y0 = x
        x_val, Y0_val = x_val
    # Losses just before training
    model.eval()
    train_epochs = [0]
    with torch.no_grad():
        if is_diff_net_:
            train_loss = [loss_func(model(x,Y0), y).item()]
            test_loss = [loss_func(model(x_val,Y0_val), y_val).item()]
        else:
            train_loss = [loss_func(model(x), y).item()]
            test_loss = [loss_func(model(x_val), y_val).item()]
    
    # Training iterations:
    for epoch in range(epochs):
        model.train()
        if is_diff_net_:
            pred = model(x,Y0)
        else:
            pred = model(x)
        loss = loss_func(pred, y)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        model.eval() # for validation
        with torch.no_grad():
            # save for plotting
            if epoch%test_intervl==0:
                if is_diff_net_:
                    train_loss.append(loss_func(model(x,Y0), y).item())
                    test_loss.append(loss_func(model(x_val,Y0_val), y_val).item())
                else:
                    train_loss.append(loss_func(model(x), y).item())
                    test_loss.append(loss_func(model(x_val), y_val).item())
                train_epochs.append(epoch)
            # print to terminal
            if epoch%(epochs//print_times)==0:
                if is_diff_net_:
                    train_loss_latest = loss_func(model(x,Y0), y).item()
                    test_loss_latest = loss_func(model(x_val,Y0_val), y_val).item()
                else:
                    train_loss_latest = loss_func(model(x), y).item()
                    test_loss_latest = loss_func(model(x_val), y_val).item()
                print('epoch {}> train_loss: {}, test_loss: {}'.format(
                    epoch,train_loss_latest,test_loss_latest))
                #print(scheduler.get_lr()) # to verify lr
        if scheduler!=None:
            # change learning rate
            scheduler.step()
    return train_epochs, train_loss, test_loss


@torch.no_grad()
def weights_init_kaiming_fanin(m):
    '''Initialize weights of the linear layers.'''
    if isinstance(m,nn.ModuleDict):
        for subm in m:
            if isinstance(subm,nn.Linear):
                nn.init.kaiming_uniform_(subm.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

@torch.no_grad()
def weights_init_kaiming_fanout(m):
    '''Initialize weights of the linear layers.'''
    if isinstance(m,nn.ModuleDict):
        for subm in m:
            if isinstance(subm,nn.Linear):
                nn.init.kaiming_uniform_(subm.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')


