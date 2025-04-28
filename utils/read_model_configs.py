from JetTagger.model.ParticleTransformer import PairEmbedConfig, EmbedConfig, BlockConfig
from JetTagger.model.MomentumCloudNet import MomentumCloudNetConfig
from typing import Dict

def read_model_configs(model:str = 'ParticleTransformer', parameters: Dict = None):
    """
    Parses and standardizes model-specific configuration dictionaries.
    
    
    Based on the model type, this function converts raw parameter dictionaries into structured configuration objects (e.g., PairEmbedding, EmbedConfig, BlockConfig)
    
    Args:
        model (str): Name of the model. Supported options are:
            - 'ParticleTransformer'
            - 'MomentumCloudNet'
            - 'LorentzNet'
            - 'ParticleNet'
            Default is 'ParticleTransformer'
        parameter (dict): Raw configuration parameters for the model.
    Returns:
        dict: A structured configuration dictionary for model initialization
    
    """
    assert parameters is not None
    configs = dict()
    
    #=================================
    # Settings for ParticleTransformer
    #=================================
    if model == 'ParticleTransformer': 
        
        #Setup pairwise embedding configuration
        configs['pairembed_config'] = PairEmbedConfig(
            pairwise_lv_dim = parameters['PairEmbed'].get('pairwise_lv_dim', None),
            pairwise_input_dim = parameters['PairEmbed'].get('pairwise_input_dim', None),
            dims = parameters['PairEmbed'].get('dims', None),
            remove_self_pair = parameters['PairEmbed'].get('remove_self_pair', None),
            use_pre_activation_pair = parameters['PairEmbed'].get('use_pre_activation_pair', None)
        ) 
        
        #Setup feature embedding configuration
        configs['embed_config'] = EmbedConfig(
            in_features = parameters['Embed']['in_features'],
            embeddims = parameters['Embed']['embeddims'],
            batch_norm_input = parameters['Embed']['batch_norm_input'],
            activation = parameters['Embed']['activation']
        )
        #Setup number of Attention and Particle-Attention Block
        configs['num_block_layers'] = parameters.get('num_block_layers', None)
        configs['num_clsblock_layers'] = parameters.get('num_clsblock_layers', None)
        
        #Setup Attention block configuration 
        configs['block_config'] = BlockConfig(
            embed_dim = parameters['block'].get('embed_dim', None),
            num_heads = parameters['block'].get('num_heads', None),
            ffn_ratio = parameters['block'].get('ffn_ratio', None),
            dropout = parameters['block'].get('dropout', None),
            attn_dropout = parameters['block'].get('attn_dropout', None),
            activation_dropout = parameters['block'].get('activation_dropout', None),
            add_bias_kv = parameters['block'].get('add_bias_kv', None),
            activation = parameters['block'].get('activation', None),
            scale_fc = parameters['block'].get('scale_fc', None),
            scale_attn = parameters['block'].get('scale_attn', None),
            scale_heads = parameters['block'].get('scale_heads', None),
            scale_resids = parameters['block'].get('scale_resids', None),
        )
        #Setup Particle Attention Block
        configs['clsblock_config'] = BlockConfig(
            embed_dim = parameters['clsblock'].get('embed_dim', None),
            num_heads = parameters['clsblock'].get('num_heads', None),
            ffn_ratio = parameters['clsblock'].get('ffn_ratio', None),
            dropout = parameters['clsblock'].get('dropout', None),
            attn_dropout = parameters['clsblock'].get('attn_dropout', None),
            activation_dropout = parameters['clsblock'].get('activation_dropout', None),
            add_bias_kv = parameters['clsblock'].get('add_bias_kv', None),
            activation = parameters['clsblock'].get('activation', None),
            scale_fc = parameters['clsblock'].get('scale_fc', None),
            scale_attn = parameters['clsblock'].get('scale_attn', None),
            scale_heads = parameters['clsblock'].get('scale_heads', None),
            scale_resids = parameters['clsblock'].get('scale_resids', None),
        )
        
        #Additional flags
        configs['use_amp'] = parameters.get('use_amp', False)
        configs['for_inference'] = parameters.get('for_inference', False)
        configs['fc_params'] = parameters.get('fc_params', None)
        
        #Check required values
        assert configs.get('num_block_layers') is not None
        assert configs.get('num_clsblock_layers')  is not None
    elif model == 'MomentumCloudNet': 
        #=================================
        # Settings for MomentumCloudNet
        #=================================
        configs = parameters    
    elif model == 'LorentzNet':
        #=================================
        # Settings for LorentzNet
        #=================================
        configs = parameters
    elif model == 'ParticleNet':
        #=================================
        # Settings for ParticleNet
        #=================================
        configs['particle_net_parameters'] = parameters 
    return configs