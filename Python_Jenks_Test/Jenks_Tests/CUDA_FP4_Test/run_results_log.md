# Unified Quant Eval — Results Log
# RETAIN_FP16=()  FULL_HADAMARD=True  WBLOCK=16  HW_BLOCK=32  CALIB_WINDOWS=32
# WikiText-2 test, non-overlapping 2048-token windows (GPTQ protocol)

## facebook/opt-125m
fp16_baseline            = 27.6559
nvfp4_w4a4_nohess        = 32.6674  (+5.0115)
w4a16_hadamard           = 32.3911  (+4.7352)
gf4_fixed_w4a4           = 37.5214  (+9.8655)
gf4_adaptive_w4a4        = 35.3440  (+7.6882)
gf4_residual2_w4a4       = 32.4087  (+4.7528)
nvfp4_acts_hess          = 38.0042  (+10.3484)
gf4_fixed_kv4            = 38.2658  (+10.6099)
gf4_adaptive_kv4         = 36.0711  (+8.4152)
hw_round_Q14             = 32.5402  (+4.8843)
hw_opt_Q14               = 32.0530  (+4.3971)
hw_opt_pop2              = 32.0920  (+4.4361)
hw_opt_Q14_kv4           = 32.7771  (+5.1212)

## facebook/opt-1.3b
fp16_baseline            = 14.6238
nvfp4_w4a4_nohess        = 18.3858  (+3.7620)
w4a16_hadamard           = 16.6422  (+2.0184)
gf4_fixed_w4a4           = 19.0133  (+4.3895)
gf4_adaptive_w4a4        = 18.3719  (+3.7481)
gf4_residual2_w4a4       = 16.7459  (+2.1221)
nvfp4_acts_hess          = 18.5873  (+3.9635)
gf4_fixed_kv4            = 20.0276  (+5.4037)
gf4_adaptive_kv4         = 19.4175  (+4.7937)
hw_round_Q14             = 18.6927  (+4.0689)
hw_opt_Q14               = 18.2145  (+3.5907)
hw_opt_pop2              = 18.3377  (+3.7139)
hw_opt_Q14_kv4           = 19.2001  (+4.5763)

## facebook/opt-2.7b
fp16_baseline            = 12.4712
nvfp4_w4a4_nohess        = 13.5837  (+1.1125)
w4a16_hadamard           = 13.5878  (+1.1166)
gf4_fixed_w4a4           = 13.9598  (+1.4886)
gf4_adaptive_w4a4        = 13.8379  (+1.3668)
gf4_residual2_w4a4       = 13.5929  (+1.1218)
nvfp4_acts_hess          = 13.8745  (+1.4034)
gf4_fixed_kv4            = 14.1791  (+1.7079)
gf4_adaptive_kv4         = 14.0784  (+1.6072)
hw_round_Q14             = 13.6002  (+1.1290)
hw_opt_Q14               = 13.5159  (+1.0447)
hw_opt_pop2              = 13.5409  (+1.0697)
hw_opt_Q14_kv4           = 13.7823  (+1.3112)

## facebook/opt-6.7b
fp16_baseline            = 10.8602
nvfp4_w4a4_nohess        = 12.4806  (+1.6204)
w4a16_hadamard           = 12.3914  (+1.5312)
gf4_fixed_w4a4           = 12.9392  (+2.0790)
gf4_adaptive_w4a4        = 12.7617  (+1.9015)
gf4_residual2_w4a4       = 12.3989  (+1.5387)
nvfp4_acts_hess          = 12.8833  (+2.0231)
gf4_fixed_kv4            = 13.1128  (+2.2526)
gf4_adaptive_kv4         = 12.9099  (+2.0497)
hw_round_Q14             = 12.4414  (+1.5812)
hw_opt_Q14               = 12.3650  (+1.5048)
hw_opt_pop2              = 12.3376  (+1.4774)
hw_opt_Q14_kv4           = 12.5171  (+1.6569)

## facebook/opt-13b
fp16_baseline            = 10.1274
nvfp4_w4a4_nohess        = 10.6517  (+0.5243)
hw_round_Q14             = 10.7866  (+0.6592)
hw_opt_Q14               = 10.9565  (+0.8291)
hw_opt_pop2              = 10.9449  (+0.8175)
hw_opt_Q14_kv4           = 11.0958  (+0.9684)
# Hessian configs: OOM'd (73 GB allocated during streaming calib, only 6 GB free on 80 GB GPU)

## facebook/opt-30b
# (pending)

## meta-llama/Llama-2-7b-hf
fp16_baseline            = 5.4721
nvfp4_w4a4_nohess        = 5.9322  (+0.4600)
w4a16_hadamard           = 5.9501  (+0.4779)
gf4_fixed_w4a4           = 6.4739  (+1.0017)
gf4_adaptive_w4a4        = 6.2824  (+0.8103)
gf4_residual2_w4a4       = 5.9592  (+0.4870)
nvfp4_acts_hess          = 6.3963  (+0.9242)
gf4_fixed_kv4            = 6.6068  (+1.1347)
gf4_adaptive_kv4         = 6.4036  (+0.9315)
hw_round_Q14             = 5.9972  (+0.5251)
hw_opt_Q14               = 5.9558  (+0.4837)
hw_opt_pop2              = 5.9622  (+0.4900)
hw_opt_Q14_kv4           = 6.0393  (+0.5672)
# NOTE: Llama-2-7B not in results_table.tex (table uses 2-13B column)

## meta-llama/Llama-2-13b-hf
fp16_baseline            = 4.8837
nvfp4_w4a4_nohess        = 5.1225  (+0.2388)
hw_round_Q14             = 5.1660  (+0.2822)
hw_opt_Q14               = 5.1803  (+0.2966)
hw_opt_pop2              = 5.1785  (+0.2948)
hw_opt_Q14_kv4           = 5.2297  (+0.3460)
w4a16_hadamard           = 5.1218  (+0.2381)
gf4_fixed_w4a4           = 5.4007  (+0.5170)
gf4_adaptive_w4a4        = 5.2784  (+0.3946)
gf4_residual2_w4a4       = 5.1251  (+0.2414)
nvfp4_acts_hess          = 5.3152  (+0.4315)
gf4_fixed_kv4            = 5.4687  (+0.5849)
gf4_adaptive_kv4         = 5.3412  (+0.4575)

## meta-llama/Meta-Llama-3-8B
fp16_baseline            = 6.1358
nvfp4_w4a4_nohess        = 7.1984  (+1.0626)
hw_round_Q14             = 7.4206  (+1.2847)
hw_opt_Q14               = 7.3130  (+1.1772)
hw_opt_pop2              = 7.3149  (+1.1790)
hw_opt_Q14_kv4           = 7.4999  (+1.3641)
w4a16_hadamard           = 7.1872  (+1.0514)
gf4_fixed_w4a4           = 7.8047  (+1.6689)
gf4_adaptive_w4a4        = 7.6121  (+1.4763)
gf4_residual2_w4a4       = 7.1929  (+1.0571)
nvfp4_acts_hess          = 7.7206  (+1.5847)
gf4_fixed_kv4            = 8.0023  (+1.8664)
gf4_adaptive_kv4         = 7.7966  (+1.6607)

## meta-llama/Llama-3.2-1B
fp16_baseline            = 9.7510
nvfp4_w4a4_nohess        = 13.1785  (+3.4275)
w4a16_hadamard           = 13.5353  (+3.7843)
gf4_fixed_w4a4           = 15.8887  (+6.1378)
gf4_adaptive_w4a4        = 15.1254  (+5.3744)
gf4_residual2_w4a4       = 13.5551  (+3.8041)
nvfp4_acts_hess          = 15.4981  (+5.7472)
gf4_fixed_kv4            = 17.5318  (+7.7808)
gf4_adaptive_kv4         = 16.6457  (+6.8947)
hw_round_Q14             = 13.6383  (+3.8873)
hw_opt_Q14               = 13.3238  (+3.5728)
hw_opt_pop2              = 13.3258  (+3.5748)
hw_opt_Q14_kv4           = 14.5220  (+4.7710)
# NOTE: Llama-3.2-1B not a table column — log only

## meta-llama/Llama-3.2-3B
fp16_baseline            = 7.8137
nvfp4_w4a4_nohess        = 9.2398  (+1.4261)
hw_round_Q14             = 9.4063  (+1.5927)
hw_opt_Q14               = 9.3028  (+1.4891)
hw_opt_pop2              = 9.3012  (+1.4875)
hw_opt_Q14_kv4           = 9.5776  (+1.7639)
# w4a16_hadamard, gf4_*, nvfp4_acts_hess, gf4_*_kv4 pending
# NOTE: Llama-3.2-3B not a table column — log only

## mistralai/Mistral-7B-v0.1
fp16_baseline            = 5.2519
nvfp4_w4a4_nohess        = 5.8946  (+0.6427)
w4a16_hadamard           = 5.5900  (+0.3381)
gf4_fixed_w4a4           = 6.0959  (+0.8440)
gf4_adaptive_w4a4        = 5.8817  (+0.6298)
gf4_residual2_w4a4       = 5.5956  (+0.3437)
nvfp4_acts_hess          = 6.0145  (+0.7626)
gf4_fixed_kv4            = 6.1958  (+0.9439)
gf4_adaptive_kv4         = 5.9708  (+0.7189)
hw_round_Q14             = 5.9578  (+0.7060)
hw_opt_Q14               = 5.8775  (+0.6256)
hw_opt_pop2              = 5.8783  (+0.6264)
hw_opt_Q14_kv4           = 5.9568  (+0.7049)

## Qwen/Qwen2.5-7B
fp16_baseline            = 6.8485
nvfp4_w4a4_nohess        = 7.6371  (+0.7886)
w4a16_hadamard           = 7.5427  (+0.6942)
gf4_fixed_w4a4           = 7.8333  (+0.9848)
gf4_adaptive_w4a4        = 7.7473  (+0.8988)
gf4_residual2_w4a4       = 7.5453  (+0.6968)
nvfp4_acts_hess          = 7.8159  (+0.9674)
gf4_fixed_kv4            = INVALID  # k_norm hook: hooks k_proj before k_norm; re-run with Cell 9b fix
gf4_adaptive_kv4         = INVALID  # same
hw_round_Q14             = 7.7653  (+0.9168)
hw_opt_Q14               = 7.6513  (+0.8028)
hw_opt_pop2              = 7.6613  (+0.8128)
hw_opt_Q14_kv4           = INVALID  # same — re-run with Cell 9b fix

## Qwen/Qwen2.5-14B
fp16_baseline            = 5.2906
nvfp4_w4a4_nohess        = 5.9902  (+0.6996)
hw_round_Q14             = 6.0897  (+0.7991)
hw_opt_Q14               = 6.0292  (+0.7386)
hw_opt_pop2              = 6.0370  (+0.7464)
# hw_opt_Q14_kv4, w4a16_hadamard, gf4_*, nvfp4_acts_hess, kv4 variants pending
