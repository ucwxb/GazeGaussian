import torch

ckp = torch.load('/bigdata/share2/xiaobao23/gaze/GazeNeRF_work_dirs/train_all_oriunet_down4_continue_eyeimport100.0_gazeimport0.0001_eyeattrimlp_2024_11_03_14_24_46/checkpoints/gaussianhead_epoch_16.pth', map_location='cpu')

ckp_new = {}

ckp_new['gazegaussian'] = ckp['gaussianhead']
ckp_new['iden_offset'] = ckp['iden_offset'][:72000]
ckp_new['expr_offset'] = ckp['expr_offset'][:72000]
ckp_new['delta_EulurAngles'] = ckp['delta_EulurAngles']
ckp_new['delta_Tvecs'] = ckp['delta_Tvecs']

torch.save(ckp_new, './gazegaussian_ckp.pth')