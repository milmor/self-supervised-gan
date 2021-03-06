hparams = {'batch_size': 8, # Default: 8
           'noise_dim': 256,
           'g_dim': 512,
           'g_learning_rate': 0.0002, # Default: 0.0002
           'g_beta_1': 0.5, # Default: 0.5
           'g_beta_2': 0.99, # Default: 0.99
           'g_initializer': 'orthogonal',
           'd_dim': 512,
           'd_learning_rate': 0.0002, # Default: 0.0002
           'd_beta_1': 0.5, # Default: 0.5
           'd_beta_2': 0.99, # Default: 0.99
           'd_initializer': 'orthogonal',
           'dec_dim': 128,
           'rec_layers': ['block1_conv2', 'block2_conv2', 'block3_conv3'],
           'rec_weight': 0.001,
           'gp_weight': 0.001,
           'd_steps': 1,
           'loss': 'hinge', # Loss types: ('bce', 'hinge', 'wgan') Default: 'hinge'
           'policy': 'color,translation'}
