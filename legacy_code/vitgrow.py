import torch
import torch.nn as nn
import numpy as np
import random
import math

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def adjust_input_embedding(old_model, new_model, mode): # D,C,P,P
    old_conv_proj = old_model.conv_proj
    new_conv_proj = new_model.conv_proj
    old_embed_dim = old_conv_proj.out_channels
    new_embed_dim = new_conv_proj.out_channels
    C = old_conv_proj.in_channels
    P = old_conv_proj.weight.size(2)
    old_config = old_model.config
    num_heads = old_config['num_heads']
    old_head_dim = old_embed_dim//num_heads
    new_head_dim = new_embed_dim//num_heads

    old_conv_weight = old_conv_proj.weight
    new_conv_weight = new_conv_proj.weight

    old_conv_weight_reshaped = old_conv_weight.view(num_heads, old_head_dim , C, P, P)
    new_conv_weight_reshaped = new_conv_weight.view(num_heads, new_head_dim , C, P, P)

    with torch.no_grad():
        new_conv_weight_reshaped[:,:old_head_dim,:,:,:] = old_conv_weight_reshaped
    
        if old_conv_proj.out_channels < new_conv_proj.out_channels:
            if mode == 4:
                for head in range(num_heads):
                    std = old_conv_weight_reshaped[head].std().item()
                    nn.init.trunc_normal_(new_conv_weight_reshaped[head, old_head_dim:, :, :, :], std=std)
                    print(f'mode {mode} conv_proj: std = {std}')
            elif mode == 5:
                nn.init.zeros_(new_conv_weight_reshaped[:, old_head_dim:, :, :, :])
                print(f'mode {mode} new conv_proj: zero')

    new_conv_proj.weight.data.copy_(new_conv_weight_reshaped.view(new_embed_dim, C, P, P))

    if old_conv_proj.bias is not None:
            old_conv_bias = old_conv_proj.bias
            new_conv_bias  = new_conv_proj.bias

            old_conv_bias_reshaped = old_conv_bias.view(num_heads, old_head_dim)
            new_conv_bias_reshaped = new_conv_bias.view(num_heads, new_head_dim)

            with torch.no_grad():
                new_conv_bias_reshaped[:,:old_head_dim] = old_conv_bias_reshaped
                nn.init.zeros_(new_conv_bias_reshaped[:,old_head_dim:])
    
    new_conv_proj.bias.data.copy_(new_conv_bias_reshaped.view(new_embed_dim))
    

def adjust_position_embedding(old_model, new_model, mode): 
    old_pos_embedding = old_model.encoder.pos_embedding # 1,N,D
    new_pos_embedding = new_model.encoder.pos_embedding
    long_seg = old_pos_embedding.shape[1]
    old_embed_dim = old_pos_embedding.shape[2]
    old_config = old_model.config
    num_heads = old_config['num_heads']
    new_config = new_model.config
    new_embed_dim = new_config['hidden_dim']
    old_head_dim = old_embed_dim//num_heads
    new_head_dim = new_embed_dim//num_heads

    old_pos_embedding_reshaped = old_pos_embedding.view(1,long_seg, num_heads, old_head_dim)
    new_pos_embedding_reshaped = new_pos_embedding.view(1,long_seg, num_heads, new_head_dim)

    with torch.no_grad():
        new_pos_embedding_reshaped[:, :, :, :old_head_dim] = old_pos_embedding_reshaped

        if mode == 1:
            nn.init.zeros_(new_pos_embedding_reshaped[:, :, :, old_head_dim:])  
            print(f'mode {mode} new pos_embedding: 0')    

        elif mode == 2:
            nn.init.zeros_(new_pos_embedding_reshaped[:, :, :, old_head_dim:])   
            print(f'mode {mode} new pos_embedding: 0')  
        
        elif mode == 4:
            for head in range(num_heads):
                std = old_pos_embedding_reshaped[:, :, head, :].std().item()
                new_pos_embedding_reshaped[:, :, head, old_head_dim:].normal_(0, std)
                print(f'mode {mode} pos_embedding: std = {std}')

        elif mode == 5:
            nn.init.zeros_(new_pos_embedding_reshaped[:, :, :, old_head_dim:])  
            print(f'mode {mode} new pos_embedding: zero')
            
    new_model.encoder.pos_embedding.data.copy_(new_pos_embedding_reshaped.view(1,long_seg,new_embed_dim))

           
    old_class_token = old_model.class_token #1,1,D
    new_class_token = new_model.class_token
    old_class_token_reshaped = old_class_token.view(1,1, num_heads, old_head_dim)
    new_class_token_reshaped = new_class_token.view(1,1, num_heads, new_head_dim)
                                                    
    with torch.no_grad(): 
        new_class_token_reshaped[:, :, :, :old_head_dim] = old_class_token_reshaped

        if mode == 1:
            nn.init.zeros_(new_class_token_reshaped[:, :, :, old_head_dim:])
            print(f'mode {mode} new class_token: 0')
        elif mode == 2:
            nn.init.zeros_(new_class_token_reshaped[:, :, :, old_head_dim:])
            print(f'mode {mode} new class_token: 0')
        
        elif mode == 4:
            for head in range(num_heads):
                std = old_class_token_reshaped[:, :, head, :].std().item()
                new_class_token_reshaped[:, :, head, old_head_dim:].normal_(0, std)
                print(f'mode {mode} class_token: std = {std}')
        elif mode == 5:
            nn.init.zeros_(new_class_token_reshaped[:, :, :, old_head_dim:])
            print(f'mode {mode} new class_token: zero')
    
    new_model.class_token.data.copy_(new_class_token_reshaped.view(1,1,new_embed_dim))
    

def adjust_classification_head(old_model, new_model, mode): #class,D
    old_head = old_model.heads.head
    new_head = new_model.heads.head
    old_head_weight = old_head.weight 
    new_head_weight = new_head.weight  
    num_class = old_head_weight.shape[0]
    old_embed_dim = old_head.in_features
    fan_in = new_head.in_features
    old_config = old_model.config
    num_heads = old_config['num_heads']
    new_config = new_model.config
    new_embed_dim = new_config['hidden_dim']
    old_head_dim = old_embed_dim//num_heads
    new_head_dim = new_embed_dim//num_heads

    old_head_weight_reshaped = old_head_weight.view(num_class, num_heads, old_head_dim)
    new_head_weight_reshaped = new_head_weight.view(num_class, num_heads, new_head_dim)


    with torch.no_grad():
        new_head_weight_reshaped[:,:,:old_head_dim] = old_head_weight_reshaped

        if mode == 4:
            for head in range(num_heads):
                std = old_head_weight_reshaped[:, head, :].std().item()
                new_head_weight_reshaped[:, head, old_head_dim:].normal_(0, std)
                print(f'mode {mode} new head.weight: std = {std}')


    new_model.heads.head.weight.data.copy_(new_head_weight_reshaped.view(num_class, new_embed_dim))

    if old_head.bias is not None:
        with torch.no_grad():
            new_head.bias = old_head.bias


def adjust_transformer_mlp(old_block,new_block, num_heads, old_embed_dim, new_embed_dim, mode): 
        old_head_dim = old_embed_dim//num_heads
        new_head_dim = new_embed_dim//num_heads

        old_linear1 = old_block.mlp[0]  
        old_linear2 = old_block.mlp[3]  

        new_linear1 = new_block.mlp[0]  
        new_linear2 = new_block.mlp[3] 

        old_mlp_dim = old_linear1.out_features
        old_weight_1 = old_linear1.weight #4D,D
        old_weight_2 = old_linear2.weight #D,4D
        new_mlp_dim = new_linear1.out_features
        new_weight_1 = new_linear1.weight
        new_weight_2 = new_linear2.weight
              
        old_mlp_head_dim = old_mlp_dim//num_heads
        new_mlp_head_dim = new_mlp_dim//num_heads

        old_weight_1_reshaped = old_weight_1.reshape(num_heads, old_mlp_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)
        old_weight_2_reshaped = old_weight_2.reshape(num_heads, old_head_dim, num_heads, old_mlp_head_dim).permute(0, 2, 1, 3)
        new_weight_1_reshaped = new_weight_1.reshape(num_heads, new_mlp_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)
        new_weight_2_reshaped = new_weight_2.reshape(num_heads, new_head_dim, num_heads, new_mlp_head_dim).permute(0, 2, 1, 3)

        with torch.no_grad():
            if mode == 5:
                nn.init.zeros_(new_weight_1_reshaped)
                nn.init.zeros_(new_weight_2_reshaped)
                print(f'mode {mode} new transformer mlp: zero')

            new_weight_1_reshaped[:, :, :old_mlp_head_dim, :old_head_dim] = old_weight_1_reshaped
            new_weight_2_reshaped[:, :, :old_head_dim, :old_mlp_head_dim] = old_weight_2_reshaped

            if mode == 1:
                nn.init.zeros_(new_weight_1_reshaped[:, :, :old_mlp_head_dim, old_head_dim:])  
                nn.init.zeros_(new_weight_2_reshaped[:, :, :old_head_dim, old_mlp_head_dim:])  
                print(f'mode {mode} new transformer mlp: ligne par defaut, colonne 0')

            elif mode == 2:
                fan_in_1 = old_embed_dim
                fan_out_1 = new_mlp_dim
                limit_1 = math.sqrt(6 / (fan_in_1 + fan_out_1))
                torch.nn.init.uniform_(new_weight_1_reshaped[:, :, old_mlp_head_dim:, :old_head_dim], -limit_1, +limit_1)
                nn.init.zeros_(new_weight_1_reshaped[:, :, :, old_head_dim:])  

                fan_in_2 = old_mlp_dim
                fan_out_2 = new_embed_dim
                limit_2 = math.sqrt(6 / (fan_in_2 + fan_out_2))
                torch.nn.init.uniform_(new_weight_1_reshaped[:, :, old_head_dim:, :old_mlp_head_dim], -limit_2, +limit_2)
                nn.init.zeros_(new_weight_2_reshaped[:, :, :, old_mlp_head_dim:]) 
                print(f'mode {mode} new transformer mlp: ligne par defaut, colonne 0')

            elif mode == 4:
                for i in range(num_heads):
                    for j in range(num_heads):
                        std_1 = old_weight_1_reshaped[i,j].std().item()
                        new_weight_1_reshaped[i, j, :, old_head_dim:].normal_(0, std_1)
                        new_weight_1_reshaped[i, j, old_mlp_head_dim:, :].normal_(0, std_1)

                        std_2 = old_weight_2_reshaped[i,j].std().item()
                        new_weight_2_reshaped[i, j, :, old_mlp_head_dim:].normal_(0, std_2)
                        new_weight_2_reshaped[i, j, old_head_dim:, :].normal_(0, std_2)

                        print(f'mode {mode} new transformer attn in_proj_weight: head {i,j},  std_1 = {std_1}, std_2 = {std_2}') 

        new_block.mlp[0].weight.data.copy_(new_weight_1_reshaped.permute(0, 2, 1, 3).reshape(new_mlp_dim,new_embed_dim))
        new_block.mlp[3].weight.data.copy_(new_weight_2_reshaped.permute(0, 2, 1, 3).reshape(new_embed_dim,new_mlp_dim))

        if new_linear1.bias is not None:
            old_bias_1 = old_linear1.bias #4D
            new_bias_1 = new_linear1.bias

            old_bias_1_reshaped = old_bias_1.view(num_heads, old_mlp_head_dim)
            new_bias_1_reshaped = new_bias_1.view(num_heads, new_mlp_head_dim)

            with torch.no_grad():
                nn.init.zeros_(new_bias_1_reshaped)
                if mode ==3:
                    nn.init.normal_(new_bias_1_reshaped[:,old_mlp_head_dim:], std=1e-6)
                new_bias_1_reshaped[:,:old_mlp_head_dim] = old_bias_1_reshaped

            new_block.mlp[0].bias.data.copy_(new_bias_1_reshaped.reshape(new_mlp_dim))
                  

        if new_linear2.bias is not None:
            old_bias_2 = old_linear2.bias #D
            new_bias_2 = new_linear2.bias 

            old_bias_2_reshaped = old_bias_2.view(num_heads, old_head_dim)
            new_bias_2_reshaped = new_bias_2.view(num_heads, new_head_dim)

            with torch.no_grad():
                nn.init.zeros_(new_bias_2_reshaped)
                if mode ==3:
                    nn.init.normal_(new_bias_2_reshaped[:,old_head_dim:], std=1e-6)
                new_bias_2_reshaped[:,:old_head_dim] = old_bias_2_reshaped

            new_block.mlp[3].bias.data.copy_(new_bias_2_reshaped.reshape(new_embed_dim)) 


def adjust_transformer_ln(old_block,new_block, old_embed_dim, new_embed_dim, mode):
    old_ln1 = old_block.ln_1
    old_ln1_weights = old_ln1.weight
    old_ln1_bias = old_ln1.bias

    new_ln1 = new_block.ln_1
    new_ln1_weights = new_ln1.weight
    new_ln1_bias = new_ln1.bias

    old_ln2 = old_block.ln_2
    old_ln2_weights = old_ln2.weight
    old_ln2_bias = old_ln2.bias

    new_ln2 = new_block.ln_2
    new_ln2_weights = new_ln2.weight
    new_ln2_bias = new_ln2.bias

    fan_in = new_embed_dim

    std_ln1 = old_ln1_weights.std().item()
    std_ln2 = old_ln2_weights.std().item()

    with torch.no_grad():
        new_ln1_weights[:old_embed_dim] = old_ln1_weights
        new_ln1_bias[:old_embed_dim] = old_ln1_bias
        nn.init.zeros_(new_ln1_bias[old_embed_dim:])   

        new_ln2_weights[:old_embed_dim] = old_ln2_weights
        new_ln2_bias[:old_embed_dim] = old_ln2_bias
        nn.init.zeros_(new_ln2_bias[old_embed_dim:])

        if mode == 1:  
            print(f'mode {mode} new encoder_layernorm weight: par defaut')
        elif mode == 2:
            print(f'mode {mode} new encoder_layernorm weight: par defaut')
        
        elif mode == 4:
            new_ln1_weights[old_embed_dim:].normal_(0, std_ln1)
            new_ln2_weights[old_embed_dim:].normal_(0, std_ln2)
            print(f'mode {mode} new encoder_layernorm weight: std_ln1 = {std_ln1}, std_ln2 = {std_ln2}') 
        elif mode == 5:
            nn.init.zeros_(new_ln1_weights[old_embed_dim:])
            nn.init.zeros_( new_ln2_weights[old_embed_dim:])
            print(f'mode {mode} new encoder_layernorm: zero')
        
def adjust_transformer_blocks_fix_num_head(old_model, new_model, mode):
    for old_block, new_block in zip(old_model.encoder.layers, new_model.encoder.layers):
        old_attention_layer = old_block.self_attention
        new_attention_layer = new_block.self_attention
        old_head_dim = old_attention_layer.head_dim
        new_head_dim = new_attention_layer.head_dim
        old_in_proj_weight = old_attention_layer.in_proj_weight
        old_out_proj_weight = old_attention_layer.out_proj.weight
        old_config = old_model.config
        old_embed_dim = old_config['hidden_dim']
        num_heads = old_config['num_heads']
        new_config = new_model.config
        new_embed_dim = new_config['hidden_dim']
        new_in_proj_weight = new_attention_layer.in_proj_weight
        new_out_proj_weight = new_attention_layer.out_proj.weight
        
        # W_q, W_k = [h×dk, D], W_v = [h×dv, D] 
        old_q_proj_weight = old_in_proj_weight[:old_embed_dim, :old_embed_dim]
        old_k_proj_weight = old_in_proj_weight[old_embed_dim:2 * old_embed_dim, :old_embed_dim]
        old_v_proj_weight = old_in_proj_weight[2 * old_embed_dim:, :old_embed_dim]

        # new_W_q = [h'×dk', D]
        new_q_proj_weight = new_in_proj_weight[:new_embed_dim, :new_embed_dim]
        new_k_proj_weight = new_in_proj_weight[new_embed_dim:2 * new_embed_dim, :new_embed_dim]
        new_v_proj_weight = new_in_proj_weight[2 * new_embed_dim:3 * new_embed_dim, :new_embed_dim]

        #reshape
        reshaped_old_q_proj_weight = old_q_proj_weight.reshape(num_heads, old_head_dim,num_heads,old_head_dim).permute(0, 2, 1, 3)
        reshaped_old_k_proj_weight = old_k_proj_weight.reshape(num_heads, old_head_dim,num_heads,old_head_dim).permute(0, 2, 1, 3)
        reshaped_old_v_proj_weight = old_v_proj_weight.reshape(num_heads, old_head_dim,num_heads,old_head_dim).permute(0, 2, 1, 3)
        reshaped_old_out_proj_weight = old_out_proj_weight.reshape(num_heads, old_head_dim,num_heads,old_head_dim).permute(0, 2, 1, 3)

        reshaped_new_q_proj_weight = new_q_proj_weight.reshape(num_heads, new_head_dim,num_heads,new_head_dim).permute(0, 2, 1, 3)
        reshaped_new_k_proj_weight = new_k_proj_weight.reshape(num_heads, new_head_dim,num_heads,new_head_dim).permute(0, 2, 1, 3)
        reshaped_new_v_proj_weight = new_v_proj_weight.reshape(num_heads, new_head_dim,num_heads,new_head_dim).permute(0, 2, 1, 3)
        reshaped_new_out_proj_weight = new_out_proj_weight.reshape(num_heads, new_head_dim,num_heads,new_head_dim).permute(0, 2, 1, 3)

        with torch.no_grad():
            if mode == 5:
                nn.init.zeros_(reshaped_new_q_proj_weight)
                nn.init.zeros_(reshaped_new_k_proj_weight)
                nn.init.zeros_(reshaped_new_v_proj_weight)
                nn.init.zeros_(reshaped_new_out_proj_weight)
                print(f'mode {mode} new reshaped_new_in_proj_weight: 0')

            reshaped_new_q_proj_weight[:, :, :old_head_dim, :old_head_dim] = reshaped_old_q_proj_weight
            reshaped_new_k_proj_weight[:, :, :old_head_dim, :old_head_dim] = reshaped_old_k_proj_weight
            reshaped_new_v_proj_weight[:, :, :old_head_dim, :old_head_dim] = reshaped_old_v_proj_weight
            reshaped_new_out_proj_weight[:, :, :old_head_dim, :old_head_dim] = reshaped_old_out_proj_weight

            if mode == 1:
                nn.init.zeros_(reshaped_new_q_proj_weight[:, :, :old_head_dim, old_head_dim:])
                nn.init.zeros_(reshaped_new_k_proj_weight[:, :, :old_head_dim, old_head_dim:])  
                nn.init.zeros_(reshaped_new_v_proj_weight[:, :, :old_head_dim, old_head_dim:])  
                nn.init.zeros_(reshaped_new_out_proj_weight[:, :, :old_head_dim, old_head_dim:]) 
                print(f'mode {mode} new reshaped_new_in_proj_weight: 0')

            elif mode == 2:
                fan_in = old_embed_dim
                fan_out = new_embed_dim
                limit = math.sqrt(6 / (fan_in + fan_out))
                nn.init.uniform_(reshaped_new_q_proj_weight[:, :, old_head_dim:, :old_head_dim], -limit, +limit)
                nn.init.zeros_(reshaped_new_q_proj_weight[:, :, :, old_head_dim:])
                nn.init.uniform_(reshaped_new_k_proj_weight[:, :, old_head_dim:, :old_head_dim], -limit, +limit)
                nn.init.zeros_(reshaped_new_k_proj_weight[:, :, :, old_head_dim:])  
                nn.init.uniform_(reshaped_new_v_proj_weight[:, :, old_head_dim:, :old_head_dim], -limit, +limit)
                nn.init.zeros_(reshaped_new_v_proj_weight[:, :, :, old_head_dim:])  
                nn.init.uniform_(reshaped_new_out_proj_weight[:, :, old_head_dim:, :old_head_dim], -limit, +limit)
                nn.init.zeros_(reshaped_new_out_proj_weight[:, :, :, old_head_dim:]) 
                print(f'mode {mode} new reshaped_new_in_proj_weight: uniform_')

            elif mode == 4:
                for i in range(num_heads):
                    for j in range(num_heads):
                        std_q = reshaped_old_q_proj_weight[i,j].std().item()
                        reshaped_new_q_proj_weight[i, j, :, old_head_dim:].normal_(0, std_q)
                        reshaped_new_q_proj_weight[i, j, old_head_dim:, :].normal_(0, std_q)

                        std_k = reshaped_old_k_proj_weight[i,j].std().item()
                        reshaped_new_k_proj_weight[i, j, :, old_head_dim:].normal_(0, std_k)
                        reshaped_new_k_proj_weight[i, j, old_head_dim:, :].normal_(0, std_k)

                        std_v = reshaped_old_v_proj_weight[i,j].std().item()
                        reshaped_new_v_proj_weight[i, j, :, old_head_dim:].normal_(0, std_v)
                        reshaped_new_v_proj_weight[i, j, old_head_dim:, :].normal_(0, std_v)

                        std_out = reshaped_old_out_proj_weight[i,j].std().item()
                        reshaped_new_out_proj_weight[i, j, :, old_head_dim:].normal_(0, std_out)
                        reshaped_new_out_proj_weight[i, j, old_head_dim:, :].normal_(0, std_out)

                        print(f'mode {mode} new transformer attn in_proj_weight: head {i,j},  std_q = {std_q}, std_k = {std_k}, std_v = {std_v}') 
            
            new_q_proj_weight = reshaped_new_q_proj_weight.permute(0, 2, 1, 3).reshape(new_embed_dim,new_embed_dim)
            new_k_proj_weight = reshaped_new_k_proj_weight.permute(0, 2, 1, 3).reshape(new_embed_dim,new_embed_dim)
            new_v_proj_weight = reshaped_new_v_proj_weight.permute(0, 2, 1, 3).reshape(new_embed_dim,new_embed_dim)
            new_out_proj_weight = reshaped_new_out_proj_weight.permute(0, 2, 1, 3).reshape(new_embed_dim,new_embed_dim)

        #new_attention_layer.in_proj_weight = nn.Parameter(torch.cat([new_q_proj_weight, new_k_proj_weight, new_v_proj_weight], dim=0)).to(dev)
        new_attention_layer.in_proj_weight.data.copy_(torch.cat([new_q_proj_weight, new_k_proj_weight, new_v_proj_weight], dim=0).to(dev))
        new_attention_layer.out_proj.weight.data.copy_(new_out_proj_weight)

        if old_attention_layer.in_proj_bias is not None:
            old_in_proj_bias = old_attention_layer.in_proj_bias
            new_in_proj_bias = new_attention_layer.in_proj_bias

            old_q_proj_bias = old_in_proj_bias[:old_embed_dim]
            old_k_proj_bias = old_in_proj_bias[old_embed_dim:2 * old_embed_dim]
            old_v_proj_bias = old_in_proj_bias[2 * old_embed_dim:]

            new_q_proj_bias = new_in_proj_bias[:new_embed_dim]
            new_k_proj_bias = new_in_proj_bias[new_embed_dim:2 * new_embed_dim]
            new_v_proj_bias = new_in_proj_bias[2 * new_embed_dim:]

            reshaped_old_q_proj_bias = old_q_proj_bias.view(num_heads, old_head_dim)
            reshaped_old_k_proj_bias = old_k_proj_bias.view(num_heads, old_head_dim)
            reshaped_old_v_proj_bias = old_v_proj_bias.view(num_heads, old_head_dim)

            reshaped_new_q_proj_bias = new_q_proj_bias.view(num_heads, new_head_dim)
            reshaped_new_k_proj_bias = new_k_proj_bias.view(num_heads, new_head_dim)
            reshaped_new_v_proj_bias = new_v_proj_bias.view(num_heads, new_head_dim)

            with torch.no_grad():

                nn.init.zeros_(reshaped_new_q_proj_bias)
                nn.init.zeros_(reshaped_new_k_proj_bias)
                nn.init.zeros_(reshaped_new_v_proj_bias)

                reshaped_new_q_proj_bias[:,:old_head_dim] = reshaped_old_q_proj_bias
                reshaped_new_k_proj_bias[:,:old_head_dim] = reshaped_old_k_proj_bias
                reshaped_new_v_proj_bias[:,:old_head_dim] = reshaped_old_v_proj_bias
               
            new_q_proj_bias = reshaped_new_q_proj_bias.view(new_embed_dim)
            new_k_proj_bias = reshaped_new_k_proj_bias.view(new_embed_dim)
            new_v_proj_bias = reshaped_new_v_proj_bias.view(new_embed_dim)
            #new_attention_layer.in_proj_bias = nn.Parameter(torch.cat([new_q_proj_bias, new_k_proj_bias, new_v_proj_bias], dim=0)).to(dev)
            new_attention_layer.in_proj_bias.data.copy_(torch.cat([new_q_proj_bias, new_k_proj_bias, new_v_proj_bias], dim=0).to(dev))
            
        if old_attention_layer.out_proj.bias is not None:
            old_out_proj_bias = old_attention_layer.out_proj.bias
            new_out_proj_bias = new_attention_layer.out_proj.bias

            reshaped_old_out_proj_bias = old_out_proj_bias.view(num_heads, old_head_dim)
            reshaped_new_out_proj_bias = new_out_proj_bias.view(num_heads, new_head_dim)
            
            with torch.no_grad():
                nn.init.zeros_(reshaped_new_out_proj_bias)
                reshaped_new_out_proj_bias[:,:old_head_dim] = reshaped_old_out_proj_bias

            new_out_proj_bias = reshaped_new_out_proj_bias.view(new_embed_dim)
            #new_attention_layer.out_proj.bias = nn.Parameter(new_out_proj_bias)
            new_attention_layer.out_proj.bias.data.copy_(new_out_proj_bias)

        adjust_transformer_mlp(old_block, new_block, num_heads, old_embed_dim, new_embed_dim, mode)
        adjust_transformer_ln(old_block,new_block,old_embed_dim, new_embed_dim, mode)


def initialize_encoder_layernorm(old_model, new_model, mode):
    old_ln = old_model.encoder.ln # D
    new_ln = new_model.encoder.ln
    old_embed_dim = old_ln.normalized_shape[0]
    weight_std = old_ln.weight.std().item()

    with torch.no_grad():
        new_ln.weight[:old_embed_dim] = old_ln.weight
        new_ln.bias[:old_embed_dim] = old_ln.bias
        nn.init.zeros_(new_ln.bias[old_embed_dim:])

        if mode == 1:
            print(f'mode {mode} new encoder_layernorm weight: par defaut')
        elif mode == 2:
            print(f'mode {mode} new encoder_layernorm weight: par defaut')
        
        elif mode == 4:
            new_ln.weight[old_embed_dim:].normal_(0, weight_std)
            print(f'mode {mode} new encoder_layernorm weight: std = {weight_std}') 
        elif mode == 5:
            nn.init.zeros_(new_ln.weight[old_embed_dim:])
            print(f'mode {mode} new encoder_layernorm weight: zero')