import torch
from model import BERTLM


model = BERTLM(25,
               hidden=256, n_layers=4, attn_heads=4,
               seq_mode='one',
               abs_position_embed=False,
               relative_attn=True,
               relative_1d=True,
               max_relative_1d_positions=10,
               relative_3d=True,
               relative_3d_vocab_size=16)

restart_file = './checkpoints/.pfam_maskedlm_r3d_306a_ep1'
model.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))

save_file = './checkpoints/.pfam_maskedlm_r3d_306a_ep1_bert'
torch.save(model.bert.state_dict(), save_file)

restart_file = './checkpoints/.pfam_maskedlm_r3d_306a_ep1_bert'
model.bert.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))


model = BERTLM(25,
               hidden=256, n_layers=4, attn_heads=4,
               seq_mode='one',
               abs_position_embed=False,
               relative_attn=True,
               relative_1d=True,
               max_relative_1d_positions=10,
               relative_3d=False,
               relative_3d_vocab_size=16)
restart_file = './checkpoints/.pfam_maskedlm_r1d_308_ep2'
model.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))

save_file = './checkpoints/.pfam_maskedlm_r1d_308_ep2_bert'
torch.save(model.bert.state_dict(), save_file)

restart_file = './checkpoints/.pfam_maskedlm_r1d_308_ep2_bert'
model.bert.load_state_dict(torch.load(restart_file, map_location=torch.device('cpu')))
